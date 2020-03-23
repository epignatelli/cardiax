# Fenton-Karma model (three currents)


This repo provides a python implementation of the FK model described in [Multiple mechanisms of spiral wave breakup in a model of cardiac electrical activity](https://aip.scitation.org/doi/10.1063/1.1504242), using [JAX](https://github.com/google/jax)


#### Performance analysis

|  framework 	|  field size  	|  iterations 	|  `timeit` 	|
|------------	|--------------	|-------------	|-----------	|
| numpy      	| (64, 64)     	| 1e3         	|           	|
| JAX (CPU)  	| (64, 64)     	| 1e3         	|           	|
| JAX (GPU)  	| (64, 64)     	| 1e3         	| 0.572s    	|
| JAX (TPU)  	| (64, 64)     	| 1e3         	| 1.13 s    	|
|            	|              	|             	|           	|
|  numpy     	| (128, 128)   	| 1e3         	|           	|
| JAX (CPU)  	| (128, 128)   	| 1e3         	|           	|
| JAX (GPU)  	| (128, 128)   	| 1e3         	| 0.609s    	|
| JAX (TPU)  	| (128, 128)   	| 1e3         	| 0.919s    	|
|            	|              	|             	|           	|
| numpy      	| (256, 256)   	| 1e3         	|           	|
| JAX (CPU)  	| (256, 256)   	| 1e3         	|           	|
| JAX (GPU)  	| (256, 256)   	| 1e3         	| 0.647s    	|
| JAX (TPU)  	| (256, 256)   	| 1e3         	| 1.27s     	|
|            	|              	|             	|           	|
| numpy      	| (512, 512)   	| 1e3         	|           	|
| JAX (CPU)  	| (512, 512)   	| 1e3         	|           	|
| JAX (GPU)  	| (512, 512)   	| 1e3         	| 0.833s    	|
| JAX (TPU)  	| (512, 512)   	| 1e3         	| 1.16s     	|
|            	|              	|             	|           	|
| numpy      	| (1024, 1024) 	| 1e3         	|           	|
| JAX (CPU)  	| (1024, 1024) 	| 1e3         	|           	|
| JAX (GPU)  	| (1024, 1024) 	| 1e3         	| 1.74s     	|
| JAX (TPU)  	| (1024, 1024) 	| 1e3         	| 2.06s     	|
