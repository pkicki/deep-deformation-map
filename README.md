# Overview
Python code for training and testing models in Tensorflow. Encoder-decoder network with two 32-x24- inputs and output with the same size (contains map of deformations). Deformations are obtained by minimizing the difference between reference and input image. Neural net is trained in unsupervised manner by minimizing the difference of latent vectors.

# Example
Compare videos paper.mp4 and paper_dnn.mp4. The second one visualizes changes in deformation of an object obtained by *trained_net* from this repo. Each frame from a sequence is an input as well as a reference frame (first from the sequence). Original datasets are available here: https://cvlab.epfl.ch/data/dsr.
