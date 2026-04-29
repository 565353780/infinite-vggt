cd ..
git clone git@github.com:565353780/camera-control.git

pip install huggingface_hub safetensors roma gradio \
  matplotlib tqdm opencv-python scipy einops trimesh \
  tensorboard viser gradio lpips hydra-core h5py \
  accelerate transformers scikit-learn gsplat evo \
  open3d

pip install numpy==1.26.1
pip install Pillow==10.3.0
pip install pyglet <2
