# RD-DPP
The code for the paper: RD-DPP: Rate-Distortion Theory meets Determinantal Point Process to Diversify Learning Data Samples

## We currently provide:
- ``` prop1.py```: Proposition 1 for different distributions:
'Gaussian','Uniform', 'Beta', 'Binomial','Exponential','Rayleigh','Poisson'

- ```main.py```: Training for Efficient with **neural collapse** (merely pursuing diversity). The function is ```RD_DPP_diveristy```.
- ```code_diversity_empirical_old``` is the function to obtain ```sdiv(x)``` with boostrapping.




** Models are modified from: https://github.com/kuangliu/pytorch-cifar. Thanks!


## results:
- semantic diversity is stored as  ```div_{run_index}.npy```
- other results ```{args.dataset_name}_{run_index}.npy```
