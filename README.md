# GGANO: A Gene Regulatory Network Inference Approach Combining Gaussian Graph Models and Neural Ordinary Differential Equations

We have annotated the code in detail to help the reader understand the code.

## Take the synthetic ten-dimensional model to illustrate the basic idea of GGANO

There are 13 files under this file, including 

5 python code files: `Data_preparation.py`, `Calculate_thro.py`, `train.py`, `prediction.py`, `train_compare.py` and `prediction_compare.py`;

2 matlab code files: `get_cov_matrix.mat` and `Solve_theta.mat`;

> We should first run `Data_preparation.py` to prepare the data for GGANO, we can get a `Data.mat` for Gaussian graph model and a `data.pickle` for Neural ODE model.

> Then we can run `get_cov_matrix.mat` to get the sample covariance matrix, `Emp_cov.mat`, and solve the theta matrix in the Gaussian graph model through `Solve_theta.mat`.

> Analyze the data `Solve_theta.mat` through `Calculate_thro.py`, and determine the final threshold and the corresponding undirected graph structure by maximizing the likelihood function. This step will produce `Undirected Graph.png`, `likehood.png`, `hist.png`, `data10_matrix.pickle`,and `hist_color_fign.png`. 

> After we run `train.py` and `train_compare.py` respectively, we can get the `Parameters_saved.pickle` and `Parameters_saved_comp.pickle`. ps: `train.py` uses the undirected graph obtained by the Gaussian graph model as a prior constraint, while `train_compare.py` has no constraints.

> Finally, we can use this trained Neural ODE model do everything we want, including inferring the structural of gene networks by `prediction.py` and `prediction.compare`.
