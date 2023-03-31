# RD-DPP
The code for the paper: RD-DPP: Rate-Distortion Theory meets Determinantal Point Process to Diversify Learning Data Samples.

## We currently provide:
- ```prop1.py```: Proposition 1 for different distributions:
'Gaussian', 'Uniform', 'Beta', 'Binomial', 'Exponential', 'Rayleigh', 'Poisson'.

- ```main_small.py```: Training logistic regression on small datasets.
- ```main_mnist.py```: Training 3 layer CNN on MNIST.
- ```main_cifar.py```: Training for EfficientNet with **neural collapse**. The function is ```RD_DPP_diveristy```.
- ```code_diversity_empirical_old``` is the function to obtain ```sdiv(x)``` with boostrapping.

## results:
- semantic diversity is stored as  ```div_{run_index}.npy```
- other results ```{args.dataset_name}_{run_index}.npy```
- ``` show_results_cifar.ipynb``` is used to summary and plot the results
- Some results are in ```EfficientNet/exp_10_2```.

### Some results as well as the ```sdiv``` on CIFAR10

<div align="center">
	<img src="https://github.com/XiwenChen-Clemson/RD-DPP/blob/main/EfficientNet_CIFAR10_10.png" alt="Editor" width="300">
</div>
<div align="center">
	<img src="https://github.com/XiwenChen-Clemson/RD-DPP/blob/main/div_EfficientNet.png" alt="Editor" width="300">
</div>




#### Models are modified from: https://github.com/kuangliu/pytorch-cifar. Thanks!



