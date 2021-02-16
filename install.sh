CUDA_VERSION=102  # cuda112 for CUDA 11.2, cuda111 for CUDA 11.1, cuda110 for CUDA 11.0, cuda102 for CUDA 10.2, and cuda101
pip install --upgrade jax jaxlib==0.1.61+cuda$CUDA_VERSION -f https://storage.googleapis.com/jax-releases/jax_releases.html/
