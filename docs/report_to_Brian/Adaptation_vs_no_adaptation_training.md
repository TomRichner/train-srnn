# Networks with multiple timescale adaptation learn faster

## Introduction

Motivating question: do networks with multiple timescale adaptation learn more quickly than those lacking adaptation?

Scenario: dynamical modeling of a vector autoregressive dataset. Physiological datasets contain multiple time series measures with unknown dynamics. We used a kinematics dataset (17 joint positions and velocities) of a simulated cheetah running in a reinforcement learning environment [@towersGymnasiumStandardInterface2026; @wawrzynskiCatLikeRobotRealTime2009]. The data were sampled at 100 Hz and split into 20 minutes for training, 3 minutes for validation, and 3 minutes for testing.

## Networks

Our network is a continuous-time recurrent neural network that optionally includes spike frequency adaptation and short-term synaptic depression [@richnerAdaptationModulatesEffective2026]. Each network was initialized as a randomly connected sparse network with a sigmoidal nonlinearity. Each network has 300 neurons, composed of 50% excitatory and 50% inhibitory neurons. The recurrent connections are sparse, with each neuron receiving and sending about 100 connections.

We compare two network variants: (1) a network with multiple timescale spike frequency adaptation (SFA) plus short-term synaptic depression (STD), and (2) a network with neither SFA nor STD, that is, no adaptation. The networks with adaptation had 32,608 trainable parameters, while those without adaptation had 32,598, just 0.03% fewer.

## Multiple timescale adaptation

Multiple timescale adaptation and short-term synaptic depression are latent variables for each neuron that adjust its bias and gain. For modeling details see [@lundstromNeuralAdaptationFractional2023; @richnerAdaptationModulatesEffective2026].

## Input and output

The input was mapped by a linear layer to 25% of the neurons, and the output was mapped from a non-overlapping 25% of neurons with another linear layer. A skip connection was included such that the recurrent network focuses on the derivative of the dynamics.

## Training

We implemented our networks in PyTorch. We trained each network with 5 random seeds. The network was trained to do one-step prediction using backpropagation through time looking back 2.5 seconds. The Adam optimizer was used with a fixed learning rate of 0.0005 after a 3 epoch warm-up. The networks were trained for 100 epochs on an L4 GPU virtual machine.

## Results

The networks with multiple timescale adaptation learned to model the kinematics dataset at a faster rate than the networks without adaptation ({fig:adaptation}). After 100 epochs, the networks with adaptation reached a test loss of 0.0154 ± 0.0022 (mean ± s.d. across the five seeds), compared with 0.0214 ± 0.0018 without adaptation, a 1.4-fold difference. Adaptation produced the lower loss for every one of the five paired seeds (one-tailed Wilcoxon signed-rank test on the paired final test losses, W = 0, p = 0.031, n = 5 pairs; median difference 0.0058). Because all five differences share the same sign, this is the smallest p-value attainable with five pairs, so it reflects the consistency of the effect rather than its size.

![Test loss versus training epoch for networks with multiple timescale adaptation (blue) and without adaptation (dark yellow). Thin lines show the five random seeds of each variant; thick lines with markers show the mean across seeds. Both axes are logarithmic, and epochs are numbered from one, so the leftmost point is the untrained network. The two variants sharing a given seed were built from the same random recurrent weight matrix, so the five comparisons are paired.](adaptation_vs_no_adaptation_test_loss.png){#fig:adaptation}
