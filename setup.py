from distutils.core import setup

setup(name='Fenton-Karma',
      version='0.1',
      description='A JAX implementation of the Fenton-Karma cardiac model',
      author='Eduardo Pignatelli',
      author_email='edu.pignatelli@gmail.com',
      url='https://github.com/epignatelli/fenton_karma_jax',
      packages=["fk", "tests", "data"],
     )