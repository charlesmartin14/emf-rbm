# emf-rbm
Extended Mean Field Restricted Boltzmann Machine

Goal:  to port the julia emf-rbm to python as a sci-kit learn RBM library

Code:  https://github.com/eric-tramel/Boltzmann.jl

https://papers.nips.cc/paper/5788-training-restricted-boltzmann-machine-via-the-thouless-anderson-palmer-free-energy.pdf

see also:
https://github.com/dfdx/Boltzmann.jl/issues/9

##Getting Started

Install Julia and check out the Bolztmann code listed above

Julia will be installed, by default, in

~/.julia

and the Boltzmann RBM package in

~/.julia/v0.4/Boltzmann

We need to run

run `julia mnist_h5.jl`

to generate the sample data sets for the notebook

##Notebook

Just run the notebook; I am current debugging and comparing the result to the julia mnistexample.jl output

##TODO

Testing and Evaluating the RBM

see: https://www.quora.com/Is-there-any-other-dataset-except-MNIST-that-is-suitable-for-RBM-DBN-Since-Many-deep-learning-libraries-and-tutorials-use-only-MNIST-dataset-for-RBM