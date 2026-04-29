import os
import sys
import gc
import importlib.machinery as _importlib_machinery
import importlib.util as _importlib_util
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF
from shutil import rmtree
from typing import Optional, List, Dict, Tuple, Union


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# infinite_vggt/Module/detector.py -> infinite_vggt/Module -> infinite_vggt -> infinite-vggt
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, 'src')
_VGGT_PARENT_ROOT = os.path.abspath(os.path.join(_PROJECT_ROOT, '..', 'vggt'))


def _ensure_external_vggt() -> None:
    """让 ``import vggt.dependency.*`` 解析到外部 ``MMRecon/vggt`` 仓库。

    ``infinite-vggt/src/vggt/`` 是含有 ``__init__.py`` 的常规包，按 Python import
    规则会屏蔽 ``MMRecon/vggt/vggt/`` 这个 PEP 420 命名空间包，仅靠调整
    ``sys.path`` 顺序无法绕开这一优先级。本函数构造一个命名空间包对象写入
    ``sys.modules['vggt']``，使得后续 ``from vggt.dependency.* import ...``
    能在 ``MMRecon/vggt/vggt/`` 下找到子模块（含 BA / VGGSfM 工具）。
    """
    external_pkg_path = os.path.join(_VGGT_PARENT_ROOT, 'vggt')
    dep_path = os.path.join(external_pkg_path, 'dependency')
    if not os.path.isdir(dep_path):
        return

    real_external = os.path.realpath(external_pkg_path)
    existing = sys.modules.get('vggt')
    if existing is not None:
        existing_paths = [
            os.path.realpath(p) for p in getattr(existing, '__path__', []) or []
        ]
        if real_external in existing_paths:
            return

    spec = _importlib_machinery.ModuleSpec('vggt', loader=None, is_package=True)
    spec.submodule_search_locations = [external_pkg_path]
    module = _importlib_util.module_from_spec(spec)
    module.__path__ = [external_pkg_path]
    sys.modules['vggt'] = module


# 在任何 ``from vggt...`` 真实 import 发生之前先注入命名空间。
_ensure_external_vggt()

# 仍把外部 vggt 仓库放到 sys.path 最前，便于其他子模块（如 vggt.dependency.*）
# 通过相对子包查找解析到正确位置。
if os.path.isdir(_VGGT_PARENT_ROOT) and _VGGT_PARENT_ROOT not in sys.path:
    sys.path.insert(0, _VGGT_PARENT_ROOT)

# 暴露 streamvggt 包，供 StreamVGGT 模型加载。注意：``infinite-vggt/src/vggt/``
# 与上面注入的命名空间同名但有 ``__init__.py``；为防止它在 ``sys.modules['vggt']``
# 被清除后重新覆盖外部包，仅在 sys.modules 已存在 vggt 命名空间时再追加 src 路径。
if os.path.isdir(_SRC_ROOT) and _SRC_ROOT not in sys.path:
    sys.path.append(_SRC_ROOT)


import pycolmap

from camera_control.Method.data import toTensor
from camera_control.Module.camera import Camera

from colmap_manage.Method.video import videoToImages

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.utils.geometry import unproject_depth_map_to_point_map

from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    pycolmap_to_batch_np_matrix,
)


# StreamVGGT 推理使用的固定方图分辨率（与 patch_size=14 对齐：518/14=37）
VGGT_FIXED_RESOLUTION = 518
# 图像加载/BA 跟踪使用的方图分辨率（与官方 demo_colmap 对齐）
BA_LOAD_RESOLUTION = 1024


def _open_rgb_pil(image_path: str) -> Image.Image:
    """读取一张图像并以 RGB PIL 返回，alpha 合到白色背景。"""
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("RGB")


def _pil_to_padded_square(
    img: Image.Image,
    target_size: int,
    to_tensor,
) -> Tuple[torch.Tensor, np.ndarray]:
    """将 RGB PIL.Image 按最长边 padding 成方图后 resize 到 target_size。

    返回 (img_tensor (3, T, T), coords [x1, y1, x2, y2, width, height])，
    coords 为原图内容在 padded 方图（target_size）中的像素边界 + 原图尺寸。
    """
    width, height = img.size
    max_dim = max(width, height)

    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    scale = target_size / max_dim

    x1 = left * scale
    y1 = top * scale
    x2 = (left + width) * scale
    y2 = (top + height) * scale
    coords = np.array([x1, y1, x2, y2, width, height])

    square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    square_img.paste(img, (left, top))
    square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

    return to_tensor(square_img), coords


def load_and_preprocess_images_square_with_source(
    image_path_list: List[str],
    target_size: int = BA_LOAD_RESOLUTION,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """与 vggt 同名工具一致：返回原图 source_images、padded 方图、original_coords。

    单次 IO 即可同时拿到：
      - source_images：每张图像原始分辨率的 HWC float [0, 1] tensor，
        与 RGBChannel.loadImage 的输入约定一致；
      - images：所有图像按最长边 padding 到 target_size 方图后的 (N, 3, T, T) tensor，
        供 BA/StreamVGGT 推理使用；
      - original_coords：(N, 6) -> [x1, y1, x2, y2, width, height]，
        用于把 VGGT 方图上的预测结果对齐到原图分辨率。
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    source_images: List[torch.Tensor] = []
    images_list: List[torch.Tensor] = []
    original_coords: List[np.ndarray] = []
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        img = _open_rgb_pil(image_path)

        source_chw = to_tensor(img)
        source_hwc = source_chw.permute(1, 2, 0).contiguous()
        source_images.append(source_hwc)

        img_tensor, coords = _pil_to_padded_square(img, target_size, to_tensor)
        images_list.append(img_tensor)
        original_coords.append(coords)

    images = torch.stack(images_list)
    coords_tensor = torch.from_numpy(np.array(original_coords)).float()

    if len(image_path_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)
        coords_tensor = coords_tensor.unsqueeze(0)

    return source_images, images, coords_tensor


def _filter_valid_indices_streamvggt(
    images: torch.Tensor,
    model: StreamVGGT,
    cos_thresh: float = 0.95,
    target_layer: int = 23,
) -> List[int]:
    """基于帧间余弦相似度筛选有效帧索引（StreamVGGT 版本）。

    与 vggt 仓库的 robust_vggt.filter_valid_indices 等价的聚类逻辑，但只调用
    StreamVGGT.aggregator 即可，无需走完整 forward。仅用于在 robust_mode 下
    决定哪些帧需要参与最终预测，因此不返回 predictions。

    Args:
        images: (N, 3, H, W) tensor，已经在目标 device。
        model: StreamVGGT 实例。
        cos_thresh: 余弦相似度阈值。
        target_layer: 用于聚类的 global block 层索引。

    Returns:
        valid_indices: 有效帧索引列表（升序）。
    """
    num_images = int(images.shape[0])
    if num_images <= 1:
        return list(range(num_images))
    if num_images == 2:
        return [0, 1]

    aggregator = model.aggregator
    patch_start_idx = aggregator.patch_start_idx

    aggregated_tokens_out: Dict[int, torch.Tensor] = {}

    def _hook(_module, _inp, out):
        aggregated_tokens_out[target_layer] = out.detach() if isinstance(out, torch.Tensor) else out

    handle = aggregator.global_blocks[target_layer].register_forward_hook(_hook)

    images_bs = images.unsqueeze(0)  # (1, S, C, H, W)
    use_cuda = images.is_cuda
    if use_cuda:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    try:
        with torch.inference_mode():
            if use_cuda:
                with torch.cuda.amp.autocast(dtype=dtype):
                    aggregator(images_bs)
            else:
                aggregator(images_bs)
    finally:
        handle.remove()

    if target_layer not in aggregated_tokens_out:
        print('[WARN][_filter_valid_indices_streamvggt]')
        print(f'\t aggregated_tokens not captured for layer {target_layer}; '
              f'returning all frames as valid.')
        return list(range(num_images))

    block_out = aggregated_tokens_out[target_layer]

    # 期望 shape: (B, S*P, C) 或 (B, S, P, C)
    if block_out.ndim == 3:
        total_tokens = block_out.shape[1]
        P = total_tokens // num_images
        global_tokens = block_out.view(1, num_images, P, block_out.shape[-1])
    elif block_out.ndim == 4:
        global_tokens = block_out
    else:
        print('[WARN][_filter_valid_indices_streamvggt]')
        print(f'\t unexpected token shape {block_out.shape}; returning all frames as valid.')
        return list(range(num_images))

    feature = global_tokens[:, :, patch_start_idx:, :].detach().float()
    frame_features = feature.mean(dim=2).squeeze(0)  # (N, C)
    frame_features_norm = F.normalize(frame_features, p=2, dim=-1)
    cos_sim_matrix = torch.mm(frame_features_norm, frame_features_norm.t())  # (N, N)

    N = cos_sim_matrix.shape[0]
    remaining = set(range(N))
    clusters: List[set] = []

    while remaining:
        remaining_list = sorted(list(remaining))

        if len(remaining) == 1:
            clusters.append(remaining.copy())
            remaining.clear()
            break

        max_sim = -1.0
        best_i, best_j = remaining_list[0], remaining_list[1]
        for i in remaining_list:
            for j in remaining_list:
                if i == j:
                    continue
                sim = cos_sim_matrix[i, j].item()
                if sim > max_sim:
                    max_sim = sim
                    best_i, best_j = i, j

        current_cluster = {best_i, best_j}
        remaining -= current_cluster
        print(f"Starting new cluster with frames {best_i} and {best_j} (similarity: {max_sim:.4f})")

        changed = True
        while changed and remaining:
            changed = False
            to_add: set = set()
            for idx in remaining:
                max_sim_to_cluster = max(cos_sim_matrix[idx, v].item() for v in current_cluster)
                if max_sim_to_cluster >= cos_thresh:
                    to_add.add(idx)
                    changed = True
            current_cluster.update(to_add)
            remaining -= to_add

        clusters.append(current_cluster)
        cluster_list = sorted(list(current_cluster))
        print(f"  Cluster {len(clusters)}: {cluster_list} ({len(current_cluster)} frames)")

    largest_cluster = max(clusters, key=len)
    valid_indices = sorted(list(largest_cluster))

    rejected_set = set(range(N)) - largest_cluster
    print(f"\nSimilarity threshold: {cos_thresh}")
    print(f"Total clusters found: {len(clusters)}")
    print(f"Valid frames: {valid_indices} ({len(valid_indices)}/{N})")
    print(f"Rejected frames: {sorted(list(rejected_set))}")

    del aggregated_tokens_out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return valid_indices


class Detector(object):
    def __init__(
        self,
        model_file_path: Optional[str] = None,
        vggsfm_model_file_path: Optional[str] = None,
        device: str = 'cuda:0',
        total_budget: int = 1200000,
    ) -> None:
        self.vggsfm_model_file_path = vggsfm_model_file_path
        self.device = device
        self.total_budget = total_budget

        self.model = StreamVGGT(total_budget=total_budget)

        if model_file_path is not None:
            self.loadModel(model_file_path, self.device)
        return

    def loadModel(
        self,
        model_file_path: str,
        device: str = 'cuda:0',
    ) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        self.device = device

        model_state_dict = torch.load(model_file_path, map_location='cpu')
        self.model.load_state_dict(model_state_dict, strict=True)
        self.model.eval()
        # 权重加载完始终保留在 CPU，仅在推理窗口内迁到 self.device，结束后立即 offload。
        self.model = self.model.to('cpu')
        self._safeEmptyCudaCache()
        return True

    @staticmethod
    def _safeEmptyCudaCache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _moveStreamVGGTToDevice(self) -> None:
        self.model = self.model.to(self.device)

    def _offloadStreamVGGTToCPU(self) -> None:
        self.model = self.model.to('cpu')
        self._safeEmptyCudaCache()

    @staticmethod
    def _imagesToFrames(images: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """将 (N, 3, H, W) 张量拆成 StreamVGGT.inference 需要的 frames 列表。

        StreamVGGT.inference 期望每个 frame 是 {"img": (1, 3, H, W)}。
        """
        return [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]

    @staticmethod
    def _stackInferenceResults(
        ress: List[Dict[str, torch.Tensor]],
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """把 StreamVGGTOutput.ress 中每帧的 dict 堆叠成 (S, ...) 的 predictions。

        每个 res 中关键 tensor 形状（B 通常为 1）：
          - pts3d_in_other_view: (B, H, W, 3)
          - conf:               (B, H, W)
          - depth:              (B, H, W, 1)
          - depth_conf:         (B, H, W)
          - camera_pose:        (B, 9)

        函数堆叠并返回 (S, ...) 形状的 GPU/CPU tensor 字典，images 字段额外添加。
        """
        all_pts3d = torch.stack([r['pts3d_in_other_view'].squeeze(0) for r in ress], dim=0)
        all_conf = torch.stack([r['conf'].squeeze(0) for r in ress], dim=0)
        all_depth = torch.stack([r['depth'].squeeze(0) for r in ress], dim=0)
        all_depth_conf = torch.stack([r['depth_conf'].squeeze(0) for r in ress], dim=0)
        all_pose = torch.stack([r['camera_pose'].squeeze(0) for r in ress], dim=0)

        return {
            'world_points': all_pts3d,
            'world_points_conf': all_conf,
            'depth': all_depth,
            'depth_conf': all_depth_conf,
            'pose_enc': all_pose,
            'images': images,
        }

    @staticmethod
    def _predictionsToNumpy(predictions: Dict) -> Dict:
        """将 predictions 中的 tensor 全部转成 numpy（bf16 先转 float32）。

        与 vggt_detect 的输出对齐：tensor 字段保留各自原本的形状（不再 squeeze batch）。
        """
        out: Dict = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu()
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                out[key] = tensor.numpy()
            else:
                out[key] = value
        return out

    @torch.no_grad()
    def _runStreamingInference(
        self,
        vggt_images: torch.Tensor,
        use_cuda: bool,
        dtype: torch.dtype,
    ) -> List[Dict[str, torch.Tensor]]:
        """以 InfiniteVGGT streaming 方式推理 ``(N, 3, H, W)`` 视频序列。

        将整段输入视为按时间排序的连续视频帧，对应 ``run_inference.py`` /
        ``demo_viser.py`` 的标准流程：

            frames = [{"img": img_i.unsqueeze(0)} for img_i in images]
            output = model.inference(frames, cache_results=True)

        ``StreamVGGT.inference`` 会在内部按帧循环、复用 ``past_key_values``
        以及 ``past_key_values_camera`` 形成的 KV cache（容量受
        ``aggregator.total_budget`` 控制），从而支撑长视频序列推理。

        Returns:
            ``StreamVGGTOutput.ress``：长度为 N 的逐帧 dict 列表（已 detach
            到 CPU），可直接喂给 ``_stackInferenceResults``。
        """
        frames = self._imagesToFrames(vggt_images)
        print(
            f"Streaming inference over {len(frames)} frames "
            "(treated as a continuous video sequence) ..."
        )
        if use_cuda:
            with torch.cuda.amp.autocast(dtype=dtype):
                output = self.model.inference(frames, cache_results=True)
        else:
            output = self.model.inference(frames, cache_results=True)
        return output.ress

    @torch.no_grad()
    def detect(
        self,
        images: torch.Tensor,
        robust_mode: bool = True,
        cos_thresh: float = 0.95,
        vggt_resolution: int = VGGT_FIXED_RESOLUTION,
    ) -> Optional[dict]:
        '''按 InfiniteVGGT streaming 流程推理一段「视频序列」。

        本函数把输入 ``images`` 视为按时间排序的连续视频帧（推荐通过
        ``detectVideoFile`` 抽 200 帧后传入），并通过
        ``StreamVGGT.inference([{"img": img.unsqueeze(0)}, ...])`` 让模型
        在内部按帧循环、复用 KV cache，从而支撑长视频推理。

        StreamVGGT 推理统一在 vggt_resolution (默认 518) 方图下进行。
        输入 images 可以是任意分辨率（推荐 1024 padded 方图），内部会先做双线性插值。
        返回的 predictions 中 depth/depth_conf/intrinsic/world_points_from_depth 等
        全部对应 vggt_resolution 尺度。

        Args:
            images: ``(N, 3, H, W)`` 张量，应为按时间顺序排列的视频帧。
            robust_mode: 仅适用于无序图像集合。对真正的视频序列建议传 False，
                否则二次筛帧 + 重推会破坏 KV cache 的连续性。
            cos_thresh: robust 聚类的余弦相似度阈值。
            vggt_resolution: StreamVGGT 推理用的方图边长。
        '''
        if images.shape[0] == 0:
            print('[ERROR][Detector::detect]')
            print("\t images are empty!")
            return None

        print(f"Input images shape: {images.shape}")

        # 与官方 demo_colmap.run_VGGT 对齐：始终把 StreamVGGT 推理输入插值到固定分辨率
        if images.shape[-2] != vggt_resolution or images.shape[-1] != vggt_resolution:
            vggt_images = F.interpolate(
                images,
                size=(vggt_resolution, vggt_resolution),
                mode='bilinear',
                align_corners=False,
            )
        else:
            vggt_images = images
        print(f"StreamVGGT inference images shape: {vggt_images.shape}")

        use_cuda = torch.cuda.is_available() and str(self.device).startswith('cuda')
        if use_cuda:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32

        num_total_images = int(vggt_images.shape[0])
        rejected_indices: List[int] = []
        rejected_extrinsics: Optional[Dict[int, np.ndarray]] = None

        # StreamVGGT 推理窗口：进入前把模型搬到 device，离开（含异常）时立即 offload 回 CPU。
        self._moveStreamVGGTToDevice()
        try:
            vggt_images = vggt_images.to(self.device)

            valid_indices: List[int] = list(range(num_total_images))

            if robust_mode and num_total_images > 1:
                print(f"Running robust mode filtering (cos_thresh={cos_thresh})...")
                valid_indices = _filter_valid_indices_streamvggt(
                    vggt_images, self.model, cos_thresh=cos_thresh,
                )
                rejected_indices = sorted(list(set(range(num_total_images)) - set(valid_indices)))
                print(f"Robust mode: {len(valid_indices)} valid frames, "
                      f"{len(rejected_indices)} rejected frames")
                print(f"Rejected frame indices: {rejected_indices}")

                if len(rejected_indices) > 0 and len(valid_indices) > 0:
                    # 警示：robust 模式的两次 streaming inference 把视频序列拆开后再
                    # 喂给 StreamVGGT.inference，会破坏「连续视频帧」的 KV cache
                    # 语义。视频路径调用方应优先使用 robust_mode=False。
                    print('[WARN][Detector::detect]')
                    print('\t robust_mode=True 会破坏视频序列的连续性，'
                          '若输入为连续视频帧建议关闭 robust_mode。')

                    # 先用所有帧推理一次，获取被剔除帧的位姿（仅做单次 streaming inference）
                    print("First pass: getting poses for all frames via streaming inference...")
                    all_ress = self._runStreamingInference(vggt_images, use_cuda, dtype)
                    all_pose_enc = torch.stack(
                        [r['camera_pose'].squeeze(0).cpu() for r in all_ress],
                        dim=0,
                    )
                    if all_pose_enc.dtype == torch.bfloat16:
                        all_pose_enc = all_pose_enc.float()
                    all_extrinsic, _ = pose_encoding_to_extri_intri(
                        all_pose_enc.unsqueeze(0), vggt_images.shape[-2:],
                    )
                    all_extrinsic_np = all_extrinsic.squeeze(0).cpu().numpy()
                    rejected_extrinsics = {idx: all_extrinsic_np[idx] for idx in rejected_indices}
                    del all_ress
                    self._safeEmptyCudaCache()

                    # 用有效帧重新推理
                    print(f"Second pass: re-inferencing with {len(valid_indices)} valid frames...")
                    valid_vggt_images = vggt_images[valid_indices]
                    valid_ress = self._runStreamingInference(valid_vggt_images, use_cuda, dtype)

                    valid_predictions = self._stackInferenceResults(
                        valid_ress, valid_vggt_images,
                    )
                    valid_pose_enc = valid_predictions['pose_enc']
                    if valid_pose_enc.dtype == torch.bfloat16:
                        valid_pose_enc = valid_pose_enc.float()
                    valid_extrinsic, valid_intrinsic = pose_encoding_to_extri_intri(
                        valid_pose_enc.unsqueeze(0), valid_vggt_images.shape[-2:],
                    )

                    valid_extrinsic_np = valid_extrinsic.squeeze(0).cpu().numpy()
                    valid_intrinsic_np = valid_intrinsic.squeeze(0).cpu().numpy()

                    full_extrinsic = np.zeros((num_total_images, 3, 4), dtype=np.float32)
                    full_intrinsic = np.zeros((num_total_images, 3, 3), dtype=np.float32)
                    for new_idx, orig_idx in enumerate(valid_indices):
                        full_extrinsic[orig_idx] = valid_extrinsic_np[new_idx]
                        full_intrinsic[orig_idx] = valid_intrinsic_np[new_idx]
                    for orig_idx in rejected_indices:
                        full_extrinsic[orig_idx] = rejected_extrinsics[orig_idx]
                        full_intrinsic[orig_idx] = valid_intrinsic_np[0]

                    valid_predictions_np = self._predictionsToNumpy(valid_predictions)
                    valid_depth = valid_predictions_np['depth']
                    valid_depth_conf = valid_predictions_np['depth_conf']
                    valid_world_points = valid_predictions_np.get('world_points')
                    valid_world_points_conf = valid_predictions_np.get('world_points_conf')

                    H, W = valid_depth.shape[1:3]

                    full_depth = np.zeros((num_total_images, H, W, 1), dtype=valid_depth.dtype)
                    full_depth_conf = np.zeros((num_total_images, H, W), dtype=valid_depth_conf.dtype)
                    valid_pred_images = valid_predictions_np['images']
                    full_images = np.zeros(
                        (num_total_images,) + valid_pred_images.shape[1:],
                        dtype=valid_pred_images.dtype,
                    )

                    for new_idx, orig_idx in enumerate(valid_indices):
                        full_depth[orig_idx] = valid_depth[new_idx]
                        full_depth_conf[orig_idx] = valid_depth_conf[new_idx]
                        full_images[orig_idx] = valid_pred_images[new_idx]

                    predictions: Dict = {}
                    predictions['depth'] = full_depth
                    predictions['depth_conf'] = full_depth_conf
                    predictions['images'] = full_images
                    predictions['extrinsic'] = full_extrinsic
                    predictions['intrinsic'] = full_intrinsic

                    if valid_world_points is not None:
                        full_world_points = np.zeros(
                            (num_total_images, H, W, 3), dtype=valid_world_points.dtype,
                        )
                        full_world_points_conf = np.zeros(
                            (num_total_images, H, W), dtype=valid_world_points_conf.dtype,
                        )
                        for new_idx, orig_idx in enumerate(valid_indices):
                            full_world_points[orig_idx] = valid_world_points[new_idx]
                            full_world_points_conf[orig_idx] = valid_world_points_conf[new_idx]
                        predictions['world_points'] = full_world_points
                        predictions['world_points_conf'] = full_world_points_conf

                    predictions['pose_enc_list'] = None

                    print("Computing world points from depth map...")
                    world_points = unproject_depth_map_to_point_map(
                        full_depth, full_extrinsic, full_intrinsic,
                    )
                    predictions['world_points_from_depth'] = world_points

                    predictions['rejected_indices'] = np.array(rejected_indices, dtype=np.int64)
                    predictions['valid_indices'] = np.array(valid_indices, dtype=np.int64)

                    return predictions

            # 正常推理（非 robust 或 robust 但未筛掉任何帧）：把整段输入视为
            # 连续视频序列，单次 StreamVGGT streaming inference 完成（参考
            # run_inference.py / demo_viser.py 的官方流程）。
            ress = self._runStreamingInference(vggt_images, use_cuda, dtype)

            stacked = self._stackInferenceResults(ress, vggt_images)

            print("Converting pose encoding to extrinsic and intrinsic matrices...")
            pose_enc = stacked['pose_enc']
            if isinstance(pose_enc, torch.Tensor) and pose_enc.dtype == torch.bfloat16:
                pose_enc = pose_enc.float()
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc.unsqueeze(0), vggt_images.shape[-2:],
            )
            stacked['extrinsic'] = extrinsic.squeeze(0)
            stacked['intrinsic'] = intrinsic.squeeze(0) if intrinsic is not None else None

            predictions = self._predictionsToNumpy(stacked)
            predictions['pose_enc_list'] = None

            print("Computing world points from depth map...")
            depth_map = predictions['depth']
            world_points = unproject_depth_map_to_point_map(
                depth_map, predictions['extrinsic'], predictions['intrinsic'],
            )
            predictions['world_points_from_depth'] = world_points

            predictions['rejected_indices'] = (
                np.array(rejected_indices, dtype=np.int64) if rejected_indices else np.array([], dtype=np.int64)
            )
            predictions['valid_indices'] = np.array(
                list(range(num_total_images)), dtype=np.int64,
            )

            return predictions
        finally:
            del vggt_images
            self._offloadStreamVGGTToCPU()

    @staticmethod
    def _cropVGGTDepthToContent(
        depth: np.ndarray,
        conf: np.ndarray,
        coords_ba: Optional[np.ndarray],
        ba_size: int,
        vggt_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        把 StreamVGGT 518x518 全幅 depth/conf 中属于 padding 的区域裁掉，
        只保留原图最长边缩放到 vggt_size 后在方图内的有效内容子区域。

        Args:
            depth: (H_vggt, W_vggt) 的深度图（StreamVGGT 518 全幅）。
            conf: (H_vggt, W_vggt) 的深度置信度（StreamVGGT 518 全幅）。
            coords_ba: (6,) [x1, y1, x2, y2, width, height]，原图内容在 BA 方图
                       (1024) 中的有效像素边界 + 原图尺寸；为 None 时不做裁剪。
            ba_size: BA 方图边长（默认 1024）。
            vggt_size: VGGT 方图边长（默认 518）。

        Returns:
            cropped_depth, cropped_conf；裁剪失败时回退到全幅。
        '''
        if coords_ba is None:
            return depth, conf

        coords_np = np.asarray(coords_ba).reshape(-1)
        if coords_np.shape[0] < 4:
            return depth, conf

        scale = float(vggt_size) / float(ba_size)
        x1 = int(round(float(coords_np[0]) * scale))
        y1 = int(round(float(coords_np[1]) * scale))
        x2 = int(round(float(coords_np[2]) * scale))
        y2 = int(round(float(coords_np[3]) * scale))

        H, W = depth.shape[:2]
        x1 = max(0, min(x1, W))
        y1 = max(0, min(y1, H))
        x2 = max(0, min(x2, W))
        y2 = max(0, min(y2, H))

        if x2 <= x1 or y2 <= y1:
            print('[WARN][Detector::_cropVGGTDepthToContent]')
            print(f'\t empty crop region (x1={x1}, y1={y1}, x2={x2}, y2={y2}); '
                  f'falling back to full {H}x{W} depth.')
            return depth, conf

        return depth[y1:y2, x1:x2], conf[y1:y2, x1:x2]

    @torch.no_grad()
    def detectImages(
        self,
        images_ba: torch.Tensor,
        source_images: List[torch.Tensor],
        original_coords_ba: Optional[Union[torch.Tensor, np.ndarray]] = None,
        robust_mode: bool = True,
        cos_thresh: float = 0.95,
    ) -> Optional[Tuple[List[Camera], Dict]]:
        '''
        Args:
            images_ba: torch.Tensor of shape (N, 3, H_ba, W_ba)，最长边 padding 到
                    BA 分辨率（默认 1024）的方图，仅用于 BA 像素坐标系跟踪。
                    StreamVGGT 推理输入由 `detect` 内部从 images_ba 双线性插值到
                    `VGGT_FIXED_RESOLUTION`（默认 518）方图得到。
            source_images: 长度为 N 的列表，每个元素为图像文件原始分辨率 RGB tensor，
                    形状 (H_i, W_i, 3) float [0, 1]，与 `Camera.loadImage` 期望一致；
                    仅用于让最终的 Camera 持有原始图像。
            original_coords_ba: (N, 6) 或 None，每一行为
                    [x1, y1, x2, y2, width, height]，前 4 个值为原图内容在 BA 方图
                    （1024）中的像素边界，后 2 个值为原图分辨率。用于把 518 StreamVGGT
                    深度裁掉 padding 后再交给 Camera。
            robust_mode: 是否启用基于帧间相似度的鲁棒筛选。
            cos_thresh: 鲁棒筛选阈值。
        '''
        if images_ba.shape[0] == 0:
            print('[WARN][Detector::detectImages]')
            print("\t images are empty!")
            return None

        if len(source_images) != images_ba.shape[0]:
            print('[ERROR][Detector::detectImages]')
            print('\t len(source_images) != images_ba.shape[0]:',
                  len(source_images), 'vs', images_ba.shape[0])
            return None

        coords_ba_np: Optional[np.ndarray] = None
        if original_coords_ba is not None:
            if isinstance(original_coords_ba, torch.Tensor):
                coords_ba_np = original_coords_ba.detach().cpu().numpy()
            else:
                coords_ba_np = np.asarray(original_coords_ba)
            if coords_ba_np.ndim == 1:
                coords_ba_np = coords_ba_np[None, :]
            if coords_ba_np.shape[0] != images_ba.shape[0]:
                print('[ERROR][Detector::detectImages]')
                print('\t original_coords_ba.shape[0] != images_ba.shape[0]:',
                      coords_ba_np.shape[0], 'vs', images_ba.shape[0])
                return None

        # StreamVGGT 推理在 518 方图，predictions 字段都对应 518 尺度
        predictions = self.detect(
            images_ba,
            robust_mode,
            cos_thresh,
        )

        if predictions is None:
            return None

        # BA 在 BA 分辨率（即输入 images_ba 的尺度，默认 1024）下进行；
        # 优化后的 1024 内参写到 predictions['ba_intrinsic']，外参写回 predictions['extrinsic']。
        optimized_predictions = self.optimizeCameraPosesByBA(
            predictions=predictions,
            images=images_ba,
        )

        if optimized_predictions is None:
            return None

        extrinsics = optimized_predictions['extrinsic']  # (N, 3, 4) BA-optimized
        ba_intrinsics = optimized_predictions['ba_intrinsic']  # (N, 3, 3) at BA resolution
        depths = optimized_predictions['depth']  # (N, H_vggt, W_vggt, 1) at VGGT resolution
        depth_conf = optimized_predictions['depth_conf']  # (N, H_vggt, W_vggt) at VGGT resolution

        depths_2d = depths.reshape(depths.shape[0], depths.shape[1], depths.shape[2])

        ba_size = int(images_ba.shape[-1])
        vggt_size = int(depths_2d.shape[-1])

        extr_dtype = extrinsics.dtype if hasattr(extrinsics, 'dtype') else np.float32

        print('start create cameras...')
        camera_list: List[Camera] = []
        for i in range(extrinsics.shape[0]):
            extrinsic_44 = np.zeros((4, 4), dtype=extr_dtype)
            extrinsic_44[:3, :4] = extrinsics[i]
            extrinsic_44[3, :] = np.array([0, 0, 0, 1], dtype=extr_dtype)

            camera = Camera.fromVGGTPose(extrinsic_44, ba_intrinsics[i], device='cpu')
            camera.image_id = f'{(i+1):06d}.png'
            camera.loadImage(source_images[i])

            coords_i = coords_ba_np[i] if coords_ba_np is not None else None
            cropped_depth, cropped_conf = Detector._cropVGGTDepthToContent(
                depths_2d[i],
                depth_conf[i],
                coords_i,
                ba_size=ba_size,
                vggt_size=vggt_size,
            )
            camera.loadDepth(cropped_depth, cropped_conf)

            camera_list.append(camera)

        return camera_list, optimized_predictions

    @torch.no_grad()
    def detectImageFiles(
        self,
        image_file_path_list: list,
        robust_mode: bool = True,
        cos_thresh: float = 0.95,
    ) -> Optional[Tuple[List[Camera], Dict]]:
        '''
        与官方 demo_colmap.py 对齐：使用 `load_and_preprocess_images_square_with_source`
        按最长边 padding 到 BA 分辨率（默认 1024）方图，避免 crop 丢失图像信息；
        StreamVGGT 推理由 `detect` 内部从 1024 方图插值到 518 方图。
        '''
        if len(image_file_path_list) == 0:
            print('[WARN][Detector::detectImageFiles]')
            print("\t images are empty!")
            return None

        print(f"Found {len(image_file_path_list)} images")

        source_images, images_ba, original_coords_ba = load_and_preprocess_images_square_with_source(
            image_file_path_list,
            target_size=BA_LOAD_RESOLUTION,
        )

        if images_ba.shape[0] == 0:
            print('[WARN][Detector::detectImageFiles]')
            print("\t images not found!")
            return None

        result = self.detectImages(
            images_ba,
            source_images,
            original_coords_ba=original_coords_ba,
            robust_mode=robust_mode,
            cos_thresh=cos_thresh,
        )

        if result is None:
            return None

        camera_list, optimized_predictions = result
        for camera, image_file_path in zip(camera_list, image_file_path_list):
            camera.image_id = os.path.basename(image_file_path)

        return camera_list, optimized_predictions

    @torch.no_grad()
    def detectImageFolder(
        self,
        image_folder_path: str,
        robust_mode: bool = True,
        cos_thresh: float = 0.95,
    ) -> Optional[Tuple[List[Camera], Dict]]:
        if not os.path.exists(image_folder_path):
            print('[ERROR][Detector::detectImageFolder]')
            print('\t image folder not exist!')
            print('\t image_folder_path:', image_folder_path)
            return None

        image_file_name_list = os.listdir(image_folder_path)
        image_file_name_list.sort()

        image_file_path_list: List[str] = []
        for image_file_name in image_file_name_list:
            if image_file_name.split('.')[-1].lower() not in ['png', 'jpg', 'jpeg']:
                continue
            image_file_path_list.append(os.path.join(image_folder_path, image_file_name))

        return self.detectImageFiles(
            image_file_path_list,
            robust_mode,
            cos_thresh,
        )

    @torch.no_grad()
    def detectVideoFile(
        self,
        video_file_path: str,
        save_image_folder_path: str,
        robust_mode: bool = False,
        cos_thresh: float = 0.95,
        target_image_num: int = 200,
    ) -> Optional[Tuple[List[Camera], Dict]]:
        '''视频入口：抽 200 帧并按 InfiniteVGGT 视频流程推理 + BA。

        流程：
          1. ``videoToImages(target_image_num=200)`` 在视频上做均匀采样，
             把帧按时间顺序写成 ``{i:06d}.jpg``，第一帧固定为视频第 0 帧。
          2. ``detectImageFolder`` 用零填充文件名 ``sort()``，保证帧仍按
             时间顺序进入 ``detect()``。
          3. ``detect()`` 通过 ``StreamVGGT.inference([{"img":...}, ...])``
             把这 200 帧当作连续视频序列推理，模型内部逐帧更新 KV cache。
          4. 上层 ``detectImages`` 继续执行 ``optimizeCameraPosesByBA`` 优化位姿。

        Args:
            robust_mode: 默认 ``False``。视频帧本身是连续序列，robust 模式的
                筛帧 + 子集二次推理会破坏 ``StreamVGGT`` 的 KV cache 连续性，
                因此视频路径不建议开启。
            target_image_num: 抽帧数量，默认 200。
        '''
        if not os.path.exists(video_file_path):
            print('[ERROR][Detector::detectVideoFile]')
            print('\t video file not exist!')
            print('\t video_file_path:', video_file_path)
            return None

        if os.path.exists(save_image_folder_path):
            rmtree(save_image_folder_path)

        os.makedirs(save_image_folder_path, exist_ok=True)

        if not videoToImages(
            video_file_path,
            save_image_folder_path,
            target_image_num=target_image_num,
            scale=1,
            print_progress=True,
        ):
            print('[ERROR][Detector::detectVideoFile]')
            print('\t videoToImages failed!')
            print('\t video_file_path:', video_file_path)
            return None

        return self.detectImageFolder(
            save_image_folder_path,
            robust_mode=robust_mode,
            cos_thresh=cos_thresh,
        )

    @torch.no_grad()
    def optimizeCameraPosesByBA(
        self,
        predictions: Dict,
        images: torch.Tensor,
        max_reproj_error: float = 8.0,
        shared_camera: bool = False,
        camera_type: str = "SIMPLE_PINHOLE",
        vis_thresh: float = 0.2,
        max_query_pts: int = 4096,
        query_frame_num: int = 8,
        fine_tracking: bool = True,
    ) -> Optional[Dict]:
        """
        使用 Bundle Adjustment 优化相机位姿。
        与官方 demo_colmap 对齐：
            - StreamVGGT 推理在 518 方图，predictions 中 depth/depth_conf/intrinsic/images/
              world_points_from_depth 全部对应 518 尺度，本函数不再覆盖这些字段。
            - BA 在 BA 输入分辨率（默认 1024）上做：图像保持 1024，内参从 518 缩放到 1024。
            - BA 优化后的外参写回到 predictions['extrinsic']；额外写入：
                * predictions['ba_intrinsic']: (N, 3, 3) BA 优化后的 1024 尺度内参，
                  供下游 `Camera.fromVGGTPose` 直接使用。BA 失败时退回到未优化的
                  scale 后内参。
                * predictions['ba_image_size']: (2,) int64，BA 分辨率 (H_ba, W_ba)。
                * predictions['points_ba'] / predictions['colors_ba']: BA 重建出的 3D 点。
        """
        if self.vggsfm_model_file_path is None:
            print('[ERROR][Detector::optimizeCameraPosesByBA]')
            print('\t please set vggsfm model file first!')
            return None

        if images.shape[0] == 0:
            print('[ERROR][Detector::optimizeCameraPosesByBA]')
            print('\t images are empty!')
            return None

        print("Starting Bundle Adjustment optimization...")

        extrinsic = predictions['extrinsic']  # (N, 3, 4)
        intrinsic = predictions['intrinsic']  # (N, 3, 3) at vggt_resolution (518)
        depth_conf = predictions['depth_conf']  # (N, H_vggt, W_vggt) at 518
        points_3d = predictions['world_points_from_depth']  # (N, H_vggt, W_vggt, 3) at 518

        ba_height, ba_width = int(images.shape[-2]), int(images.shape[-1])
        vggt_height, vggt_width = int(depth_conf.shape[-2]), int(depth_conf.shape[-1])

        scale_w = ba_width / float(vggt_width)
        scale_h = ba_height / float(vggt_height)
        if abs(scale_w - scale_h) > 1e-6:
            print(f"[WARN][Detector::optimizeCameraPosesByBA] non-square BA scale "
                  f"(scale_w={scale_w}, scale_h={scale_h}); using anisotropic intrinsic scaling.")

        ba_intrinsic = intrinsic.astype(np.float32, copy=True)
        ba_intrinsic[:, 0, :] *= scale_w  # fx, cx
        ba_intrinsic[:, 1, :] *= scale_h  # fy, cy

        image_size = np.array([ba_height, ba_width])

        use_cuda = torch.cuda.is_available() and str(self.device).startswith('cuda')
        if use_cuda:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32

        images_gpu = toTensor(images, dtype=torch.float32, device=self.device)
        try:
            print("Predicting tracks for Bundle Adjustment...")
            if use_cuda:
                with torch.cuda.amp.autocast(dtype=dtype):
                    pred_tracks, pred_vis_scores, pred_confs, points_3d_track, points_rgb = predict_tracks(
                        self.vggsfm_model_file_path,
                        images_gpu,
                        conf=depth_conf,
                        points_3d=points_3d,
                        masks=None,
                        max_query_pts=max_query_pts,
                        query_frame_num=query_frame_num,
                        keypoint_extractor="aliked+sp",
                        fine_tracking=fine_tracking,
                    )
            else:
                pred_tracks, pred_vis_scores, pred_confs, points_3d_track, points_rgb = predict_tracks(
                    self.vggsfm_model_file_path,
                    images_gpu,
                    conf=depth_conf,
                    points_3d=points_3d,
                    masks=None,
                    max_query_pts=max_query_pts,
                    query_frame_num=query_frame_num,
                    keypoint_extractor="aliked+sp",
                    fine_tracking=fine_tracking,
                )
        finally:
            del images_gpu
            self._safeEmptyCudaCache()

        track_mask = pred_vis_scores > vis_thresh

        print("Building COLMAP reconstruction...")
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d_track,
            extrinsic,
            ba_intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=max_reproj_error,
            shared_camera=shared_camera,
            camera_type=camera_type,
            points_rgb=points_rgb,
        )

        # 不修改 predictions 中已有的 518 尺度字段；只附加 BA 相关字段。
        optimized_predictions = predictions.copy()
        optimized_predictions['ba_image_size'] = np.array([ba_height, ba_width], dtype=np.int64)

        if reconstruction is None:
            print('[ERROR][Detector::optimizeCameraPosesByBA]')
            print('\t No reconstruction can be built with BA, returning predictions without BA optimization')
            optimized_predictions['ba_intrinsic'] = ba_intrinsic
            return optimized_predictions

        print("Running Bundle Adjustment...")
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        print("Extracting optimized poses from reconstruction...")
        _, optimized_extrinsic_44, optimized_intrinsic, _ = pycolmap_to_batch_np_matrix(
            reconstruction, device="cpu", camera_type=camera_type,
        )

        if optimized_extrinsic_44.shape[-1] == 4 and optimized_extrinsic_44.shape[-2] == 4:
            optimized_extrinsic = optimized_extrinsic_44[:, :3, :]  # (N, 3, 4)
        else:
            optimized_extrinsic = optimized_extrinsic_44

        optimized_predictions['extrinsic'] = optimized_extrinsic.astype(np.float32, copy=False)
        optimized_predictions['ba_intrinsic'] = optimized_intrinsic.astype(np.float32, copy=False)

        optimized_predictions['points_ba'] = points_3d_track
        optimized_predictions['colors_ba'] = points_rgb

        print("Bundle Adjustment optimization completed successfully!")
        return optimized_predictions
