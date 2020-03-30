![build](https://github.com/epignatelli/fenton_karma_jax/workflows/build/badge.svg)

# Fenton-Karma model (three currents)


This repo provides a python implementation of the FK model described in [Multiple mechanisms of spiral wave breakup in a model of cardiac electrical activity](https://aip.scitation.org/doi/10.1063/1.1504242), using [JAX](https://github.com/google/jax)


#### Performance analysis

We test performance and scalability against two quantities, tissue size and number of iterations.
Tests are performed using the notebooks in the repository, google cola.
Time is measured using the `timeit` module over 10 runs. Result is the average.

The number of iterations is set to `1e3`.
|  framework/field size 	|  (64, 64) 	|  (128, 128) 	| (256, 256) 	| (512, 512) 	| (1024, 1024) 	|
|-----------------------	|-----------	|-------------	|------------	|------------	|--------------	|
| numpy                 	| 695 ms    	| 1.85 s     	| 9.56 s    	|     	|        	|
| JAX (CPU)             	| 336 ms    	| 904 ms      	| 2.94 s     	| 11.1 s     	| 45 s         	|
| JAX (GPU)             	| 193 ms    	| 189 ms      	| 199 ms     	| 237 ms     	| 613 ms       	|
| JAX (TPU)             	| 59 ms     	| 74 ms      	| 119 ms     	| 272 ms      	| 842 ms       	|


The field size is set to `(128, 128)`
|  framework/iterations 	| 1e2       	| 1e3         	| 1e4        	| 1e5        	| 1e6          	|
|-----------------------	|-----------	|-------------	|------------	|------------	|--------------	|
| numpy                 	| 528 ms    	| 692 ms      	| 2.35 s     	| 19.1 s     	| 3min 7s      	|
| JAX (CPU)             	| 514 ms    	| 600 ms      	| 1.47 s     	| 10.2 s     	| 1min 38s     	|
| JAX (GPU)             	| 643 ms    	| 661 ms      	| 822 ms     	| 2.19 s     	| 16.2 s       	|
| JAX (TPU)             	| 1 s       	| 782 ms      	| 992 ms     	| 1.43 s     	| 7.29 s       	|


Contributions of the vectorization of the stepping scheme `jax.lax.fori_loop` on `1e3` iterations and field size `(128, 128)`. Note that this test does not include results plotting.
|  framework/vectorization 	| compiled-for  	| uncompiled-for   	| 
|-----------------------	|----------------	|------------------	|
| numpy                 	| \             	| \                	|
| JAX (CPU)             	| 895 s    	        | 1.87 ms           | 
| JAX (GPU)             	| 186 ms        	| 2.29 s           	|
| JAX (TPU)             	| 74 ms           	| 6.46 s            |

The hardware used is as follows:

1. `lscpu` returned:
    
```
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              2
On-line CPU(s) list: 0,1
Thread(s) per core:  2
Core(s) per socket:  1
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               79
Model name:          Intel(R) Xeon(R) CPU @ 2.20GHz
Stepping:            0
CPU MHz:             2200.000
BogoMIPS:            4400.00
Hypervisor vendor:   KVM
Virtualization type: full
L1d cache:           32K
L1i cache:           32K
L2 cache:            256K
L3 cache:            56320K
NUMA node0 CPU(s):   0,1
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm rdseed adx smap xsaveopt arat md_clear arch_capabilities
```

2. `nvidia-smi` returned:

```
Wed Mar 25 13:58:37 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.64.00    Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   60C    P8    10W /  70W |      0MiB / 15079MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
