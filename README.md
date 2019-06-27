# DNN-for-Dual-Readout-Calorimetry

## Abstract
This repo contains work done with the Texas Tech group for the CMS collaboration at CERN, specifically with Dr. Federico De Guio. It is a brief study of the feasibility of using machine learning techniques, specifically neural networks for event-by-event discrimination between Cerenkov and Scintillation radiation in signals from high energy calorimeters that use the single-channel dual readout calorimetry technique. We show that machine learning has the potential to validate single-channel calorimetry as a feasible technique in future high energy particle detector calorimeterss. Moreover there are possible latency advantages if these neural networks are hardwired into FPGAs, which could aid a Level 1 trigger. 


## Table of Contents
* Introduction
* Part I: Feasibility on Monte-Carlo Simulated Data
  * Data Generation
  * Model Architecture
  * Varying Photoelectron Count
  * Varying Digitizer Frequency (Input Layer)
  * Conclusions

* Part II: Feasibility on Real Data
  * Preliminary Results
  * Future Work
  
* Part III: Implementation on FPGAs
  * hls4ml
  * Latency Predictions

* Conclusions and Future Work

* References



## Introduction
Over the last twenty years, Dual Readout Calorimetry has emerged as a possible technique for measuring the energy of hadronic jets that offer certain advantages over other calorimetry techniques. (see [1,2] for a more extensive review of the work done recently). In particular, dual readout calorimetry allows us to obtain the electromagnetic portion of the hadronic energy on an event-by-event basis, due to the ability to collect both Cerenkov and Scintillation radiation. 

*Single-channel* dual readout calorimetry entails combining both of these radiation types into one single channel. For example, a calorimeter with a combination of clear and scintillating fibers might only have one silicon photomultiplier collecting the radiation data of both types of radiation. Since these two different types of radiation have different pulse signatures, it is possible that the two types of radiation could be analyzed separately after data collection, which would potentially cut overhead costs by nearly a half. Although a fitting procedure could be utilized, it is a time-consuming procedure. This project focuses on the possibility of using neural networks to conduct this event-by-event analysis as they can be used for fast inference after training has been completed.



## Part I: Feasibility on Monte-Carlo Simulated Data
  ### Data Generation
  We model a sample pulse as below. See the file labeled pulse_library_generation.py for details. Note that this data generation was done using the HTCondor cluster at CERN.
  
  
  ### Model Architecture
  Input layer is generated pulse (100 nodes)
  Intermediate layer of 10 nodes
  Output layer is 2 nodes 
    Predict fraction of Cerenkov radiation and fraction of Scintillation radiation

  ### Varying Photoelectron Count
  ### Varying Digitizer Frequency (Input Layer)
  ### Conclusions
  Major dependence is on ratio and not scintillation decay time
  Higher photoelectron count gives better results 
  Gradual change over 1k-5k photoelectrons range (no tipping point)
  Higher digitizer freq ≠ better prediction
  Low freq hides the effects of fluctuations
  Scintillation prediction is much more stable than Cerenkov prediction

  
## Part II: Feasbility on Real Data
  ### Preliminary Results
  With 5,000 photoelectrons and 100 bins, possibility of less than 1% error
  ### Future Work
  
## Part III: Implementation on FPGAs
  ### hls4ml
  hls4ml is a package for fast inference on FPGAs. [3]
  Takes NN and creates HLS implantation that provides resource usage estimates

  Up to 6,000 parallel operations=# of multiplication units=max # of nodes and weights in NN if we aim for one clock cycle per analysis
    This for latest tech in FPGAs
    
  Can trade latency for lower resource usage

  ### Latency Predictions

## Conclusions and Future Work

## References
[3] J. Duarte et al., "Fast inference of deep neural networks in FPGAs for particle physics", JINST 13 P07027 (2018), arXiv:1804.06913.
