# RD-DPP
The code for the paper: RD-DPP: Rate-Distortion Theory meets Determinantal Point Process to Diversify Learning Data Samples.

## We currently provide:

- ```main_mnist.py```: Train 3 layer CNN on MNIST.
- ```main_small.py```: Train logistic regression on small datasets.
- ```main_cifar.py```: Train for EfficientNet on CIFAR10. 


## show results:
- ```show_results_mnist.ipynb```. Some results are in ```exp2_MNIST```.
- ```show_results_small.ipynb```. Some results are in ```exp_small_dataset```.
- ```show_results_cifar2.ipynb```. Some results are in ```exp_up_10_efficientnet```.



### Some results on CIFAR10 (Please View ```threenet_result.png.png```)

<div align="center">
	<img src="https://github.com/XiwenChen-Clemson/RD-DPP/blob/main/threenet_result.png" alt="Editor" width="800">
</div>





#### DL Models are modified from: https://github.com/kuangliu/pytorch-cifar. Thanks!
#### K-center coreset is from: https://github.com/google/active-learning/tree/master. Thanks!


