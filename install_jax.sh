CUDA_VERSION=$(nvidia-smi | grep -o 'CUDA Version: [0-9].\.[0-9]' | sed 's/.*: //' | sed 's/\.//')
pip install --upgrade jax jaxlib==0.1.64+cuda$CUDA_VERSION -f https://storage.googleapis.com/jax-releases/jax_releases.html/
