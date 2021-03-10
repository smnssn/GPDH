# Probabilistic Modeling of Thermal Grids using Gaussian Processes

Julia implementation of the method described in the paper [Probabilistic Modeling of Thermal Grids using Gaussian Processes](https://ieeexplore.ieee.org/document/9304284) by Johan Simonsson, Khalid Atta and Wolfgang Birk, published at 2020 59th IEEE Conference on Decision and Control (CDC). 

# Paper abstract 
Dynamic physics based modeling of district heating networks has gained importance due to an increased use of renewable energy sources and a transition towards lower temperature district heating networks. The modeling is enhanced by technologies for automatic model generation and co-simulation. These models are in general not suitable for automatic control and optimization methods, due to the complexity of the model. Moreover, there is no notion of uncertainty in the models, something that can be of importance for decision making, and that can be explicitly accounted for in e.g Bayesian Optimization and Stochastic Nonlinear Model Predictive Control. In this paper a data driven Gaussian process model for the thermal dynamics of the district heating grid is proposed, with a kernel derived using known physics and numerical methods. The model is trained and validated on a realistic first principle simulation model of a district heating pipe. Results show a good correspondence with the output from the training model on a validation dataset, providing explicit propagation of the input uncertainties. It is suggested that the method can be scaled up to larger parts of the grid for use in advanced control and optimization methods.

# Installation instructions:
1. Clone from Github:

    ```git clone git@github.com:smnssn/gpdh.git```

2. Run in Julia REPL: 

    ```include("Main.jl")```

# Citation
If you use this code, please cite:

  ```
  @INPROCEEDINGS{9304284,
  author={J. {Simonsson} and K. T. {Atta} and W. {Birk}},
  title={Probabilistic Modeling of Thermal Grids using Gaussian Processes}, 
  booktitle={2020 59th IEEE Conference on Decision and Control (CDC)}, 
  year={2020},
  volume={},
  number={},
  pages={36-41},
  doi={10.1109/CDC42340.2020.9304284}
  }
  ```

Note that the code is provided as a means to verify the method, and is not optimized with regards to e.g computational performance. If you want to contact me, feel free to send a mail to <johan.simonsson@ltu.se>.
