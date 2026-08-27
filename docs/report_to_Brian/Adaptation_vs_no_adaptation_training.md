# Networks with multiple timescale adaptation learn faster  

## Introduction
Motivating question: Do networks with multiple timescale adaptation learn more quickly than those lacking adaptation?

Scenario: Dynamical modeling of a vector autoregressive dataset. Physiological datasets contain multiple time series measures that have unknown dynamics.  We used a kinematics dataset (17 joint positions and velocities) of a simulated cheetah running in a renforcement learning environment (cite Gym). The data were sampled at 100 Hz and split into 20 minutes for training, 3 minutes for validation, and 3 minutes for testing.  

## Networks

Our network is a continuous-time recurrent neural network that optionally includes spike frequency adaptation and short-term synaptic depression [@richnerAdaptationModulatesEffective2026].  The networks initiated as a randomly connected sparse network with a sigmoidal nonlinearity.  Each network has 300 neurons composed of 50% excitatory and 50% inhibitory neurons.  The recurrent connections are sparse with each neuron receiving and sending about 100 connections.

We compare two network variants: (1) that has multiple timescale spike frequency adaptation (SFA) plus short-term synaptic depression (STD) and (2) a network with neither SFA nor STD, no adaptation.  The networks without adaptation had 32,608 trainable parameters, while the networks with adpation had 32,598, just 0.03% more.

## Multiple timescale adaptation

Multiple timescale adaptation and short-term synaptic depression are latent variables for each neuron which adjust its bias and gain.  For modeling details see [@lundstromNeuralAdaptationFractional2023; @richnerAdaptationModulatesEffective2026].  

## Input and output

The input was mapped by a linear layer to 25% of the neurons, and the output was mapped from a non-overlapping 25% of neurons with another linear layer. A skip connection was included such that the recurrent network focuses on the derivative of the dynamics.

## Training

We implemented our networks in PyTorch.  We trained each network with 5 random seeds.  The network was trained to do one-step prediction using backpropagation through time looking back 5 seconds.  The ADAM optimizer was used and a fixed learning rate of 0.0005 was used after a 3 epoch warm up. The networks were trained for 100 epochs on an L4 GPU virtual machine.

## Results

The networks with multiple timescale adaption learned to model the kinematics dataset at a faster rate than the model without adapation.