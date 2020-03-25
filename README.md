# Fenton-Karma model (three currents)


This repo provides a python implementation of the FK model described in [Multiple mechanisms of spiral wave breakup in a model of cardiac electrical activity](https://aip.scitation.org/doi/10.1063/1.1504242), using [JAX](https://github.com/google/jax)


#### Performance analysis

We test performance and scalability against two quantities, tissue size and number of iterations.
Tests are performed using the notebooks in the repository, google cola.
Time is measured using the `timeit` module over 10 runs. Result is the average.


The number of iterations is set to `1e2`.
|  framework/field size 	|  (64, 64) 	|  (128, 128) 	| (256, 256) 	| (512, 512) 	| (1024, 1024) 	|
|-----------------------	|-----------	|-------------	|------------	|------------	|--------------	|
| numpy                 	| 583 ms    	| 706 ms      	| 1.44 s     	| 4.42 s     	| 15.6 s       	|
| JAX (CPU)             	| 534 ms    	| 607 ms      	| 836 ms     	| 1.76 s     	| 5.39 s       	|
| JAX (GPU)             	| 659 ms    	| 667 ms      	| 679 ms     	| 766 ms     	| 994 ms       	|
| JAX (TPU)             	| 0.883s    	| 0.984s      	| 0.950s     	| 1.05s      	| 1.39s        	|


The field size is set to `(128, 128)`
|  framework/iterations 	| 1e2       	| 1e3         	| 1e4        	| 1e5        	| 1e6          	|
|-----------------------	|-----------	|-------------	|------------	|------------	|--------------	|
|                       	|           	|             	|            	|            	|              	|
|                       	|           	|             	|            	|            	|              	|
|                       	|           	|             	|            	|            	|              	|
|                       	|           	|             	|            	|            	|              	|
