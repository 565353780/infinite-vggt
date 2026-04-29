import os
import sys
sys.path.append('../camera-control')
sys.path.append('../colmap-manage')
sys.path.append('../vggt')

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

from shutil import rmtree

from camera_control.Method.pcd import toPcd
from camera_control.Module.camera_convertor import CameraConvertor

from colmap_manage.Module.colmap_renderer import COLMAPRenderer

from infinite_vggt.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    work_space = '/nvme1pnt/lichanghao/'
    model_file_path = home + '/chLi/Model/StreamVGGT/checkpoints.pth'
    vggsfm_model_file_path = home + '/chLi/Model/VGGT/vggsfm_v2_tracker.pt'
    device = 'cuda:0'
    test_folder_path = f'{work_space}/chLi/MMVideoReconV1/lichanghao/20260426_103340_362250/'
    video_file_path = test_folder_path + 'input_video.mov'
    image_folder_path = test_folder_path + 'infinite_vggt_test/input/'
    save_folder_path = test_folder_path + 'infinite_vggt_test/'
    # 视频帧是连续序列，关闭 robust_mode，保留 InfiniteVGGT 的 KV cache 连续性。
    robust_mode = False
    cos_thresh = 0.95
    total_budget = 1200000
    # 是否启用 VGGSfM + COLMAP BA。关闭时 Detector 会用 VGGT depth 反投点 +
    # 输入图像生成兜底的 ``points_ba`` / ``colors_ba``，下游点云导出仍可工作。
    is_ba_optimize = True

    detector = Detector(
        model_file_path,
        vggsfm_model_file_path,
        device,
        total_budget=total_budget,
    )

    result = detector.detectVideoFile(
        video_file_path,
        image_folder_path,
        robust_mode,
        cos_thresh,
        target_image_num=72,
        is_ba_optimize=is_ba_optimize,
    )

    assert result is not None

    camera_list, predictions = result

    print('camera num:', len(camera_list))

    pcd = toPcd(predictions['points_ba'], predictions['colors_ba'])

    save_colmap_folder_path = save_folder_path + 'colmap/'
    if os.path.exists(save_colmap_folder_path):
        rmtree(save_colmap_folder_path)

    CameraConvertor.createColmapDataFolder(
        cameras=camera_list,
        pcd=pcd,
        save_data_folder_path=save_colmap_folder_path,
        point_num_max=200000,
    )

    COLMAPRenderer.renderColmap(save_colmap_folder_path)
    return True
