# RD-DPP
The code for the paper: RD-DPP: Rate-Distortion Theory meets Determinantal Point Process to Diversify Learning Data Samples.

## We currently provide:
- ```prop1.py```: Proposition 1 for different distributions:
'Gaussian', 'Uniform', 'Beta', 'Binomial', 'Exponential', 'Rayleigh', 'Poisson'.
- ```main_mnist.py```: Train 3 layer CNN on MNIST.
- ```main_small.py```: Train logistic regression on small datasets.
- ```main_cifar.py```: Train for EfficientNet on CIFAR10. 


## show results:
- ```show_results_mnist.ipynb```. Some results are in ```exp2_MNIST```.
- ```show_results_small.ipynb```. Some results are in ```exp_small_dataset```.
- ```show_results_cifar.ipynb```. Some results are in ```EfficientNet/exp_10_2```.


## Exemplary Results
### Proposition 1 for different distribution
<div align="center">
	<img src="https://github.com/XiwenChen-Clemson/RD-DPP/blob/main/view_phase.jpg" alt="Editor" width="800">
</div>


### Some results on MNIST
<div align="center">
	<img src="https://github.com/XiwenChen-Clemson/RD-DPP/blob/main/MNIST_clean.png" alt="Editor" width="300">
</div>

### Some results on CIFAR10

<div align="center">
	<img src="https://github.com/XiwenChen-Clemson/RD-DPP/blob/main/EfficientNet_CIFAR10_10.png" alt="Editor" width="300">
</div>





#### Models are modified from: https://github.com/kuangliu/pytorch-cifar. Thanks!



