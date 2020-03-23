# Fenton-Karma model (three currents)


This repo provides a python implementation of the model described in [Multiple mechanisms of spiral wave breakup in a model of cardiac electrical activity ](https://aip.scitation.org/doi/10.1063/1.1504242), using [JAX](https://github.com/google/jax)


#### Performance analysis

|  framework 	|  Field size 	|  User time 	| Sys time 	| Total   	| **Wall time** 	|
|------------	|-------------	|------------	|----------	|---------	|---------------	|
|  numpy     	| (128, 128)  	| 3min 8s    	| 17ms     	| 3min 8s 	| **3min 9s**   	|
| JAX (CPU)  	| (128, 128)  	| 2min 4s    	| 2.55s    	| 2min 7s 	| **1min 39s**  	|
| JAX (GPU)  	| (128, 128)  	| 13.1s      	| 582ms    	| 13.7s   	| **13.7s**     	|
| JAX (TPU)  	| (128, 128)  	| 2.52s      	| 2.72s    	| 5.23s   	| **8.49s**     	|
