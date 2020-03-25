# Fenton-Karma model (three currents)


This repo provides a python implementation of the FK model described in [Multiple mechanisms of spiral wave breakup in a model of cardiac electrical activity](https://aip.scitation.org/doi/10.1063/1.1504242), using [JAX](https://github.com/google/jax)


#### Performance analysis

|  framework/field size 	|  (64, 64) 	|  (128, 128) 	| (256, 256) 	| (512, 512) 	| (1024, 1024) 	|
|-----------------------	|-----------	|-------------	|------------	|------------	|--------------	|
| numpy                 	|           	|             	|            	|            	|              	|
| JAX (CPU)             	|           	|             	|            	|            	|              	|
| JAX (GPU)             	| 0.572s    	| 0.609s      	| 0.647s     	| 0.833s     	| 1.74s        	|
| JAX (TPU)             	| 1.13 s    	| 0.919s      	| 1.27s      	| 1.16s      	| 2.06s        	|


|  framework/iterations 	| 1e1       	| 1e2         	| 1e3        	| 1e4        	| 1e5          	|
|-----------------------	|-----------	|-------------	|------------	|------------	|--------------	|
|                       	|           	|             	|            	|            	|              	|
|                       	|           	|             	|            	|            	|              	|
|                       	|           	|             	|            	|            	|              	|
|                       	|           	|             	|            	|            	|              	|
