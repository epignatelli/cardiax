from distutils.core import setup

setup(name='rde',
      version='0.1',
      description='A JAX implementation of various reaction-diffusion models',
      author='Eduardo Pignatelli',
      author_email='edu.pignatelli@gmail.com',
      url='https://github.com/epignatelli/fenton_karma_jax',
      packages=["fenton_karma", "fitzhugh_nagumo", "tests", "deepexcite"],
     )