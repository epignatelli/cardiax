# Fenton-Karma model (three currents)


This repo provides a python implementation of the FK model described in [Multiple mechanisms of spiral wave breakup in a model of cardiac electrical activity](https://aip.scitation.org/doi/10.1063/1.1504242), using [JAX](https://github.com/google/jax)


#### Performance analysis

|  framework 	|  field size 	|  iterations 	|  user time 	|  sys time 	|  total   	| **wall time** 	|
|------------	|-------------	|-------------	|------------	|-----------	|----------	|---------------	|
| numpy      	| (64, 64)    	| 100000      	| 1min 14s   	| 15ms      	| 1min 14s 	| **1min 14s**  	|
| JAX (CPU)  	| (64, 64)    	| 100000      	| 48.8s      	| 2.5s      	| 51.3s    	| **36.3s**     	|
| JAX (GPU)  	| (64, 64)    	| 100000      	| 14.2s      	| 277ms     	| 14.5s    	| **15s**       	|
| JAX (TPU)  	| (64, 64)    	| 100000      	| 1.98s      	| 2.1s      	| 4.09s    	| **6.35s**     	|
|            	|             	|             	|            	|           	|          	|               	|
|  numpy     	| (128, 128)  	| 100000      	| 3min 8s    	| 17ms      	| 3min 8s  	| **3min 9s**   	|
| JAX (CPU)  	| (128, 128)  	| 100000      	| 2min 4s    	| 2.55s     	| 2min 7s  	| **1min 39s**  	|
| JAX (GPU)  	| (128, 128)  	| 100000      	| 13.1s      	| 582ms     	| 13.7s    	| **13.7s**     	|
| JAX (TPU)  	| (128, 128)  	| 100000      	| 2.52s      	| 2.72s     	| 5.23s    	| **8.49s**     	|
|            	|             	|             	|            	|           	|          	|               	|
| numpy      	| (256, 256)  	| 100000      	| 15min 34s  	| 34.6s     	| 16min 9s 	| **16min 11s** 	|
| JAX (CPU)  	| (256, 256)  	| 100000      	| 9min 30s   	| 6.98s     	| 9min 37s 	| **5min 11s**   	|
| JAX (GPU)  	| (256, 256)  	| 100000      	| 14.4s      	| 563ms     	| 14.9s    	| **15.1s**      	|
| JAX (TPU)  	| (256, 256)  	| 100000      	| 3.03s      	| 4.36s     	| 7.37s    	| **13.1s**      	|
