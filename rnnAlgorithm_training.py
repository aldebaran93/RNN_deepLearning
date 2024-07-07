"""
Script for implementing the RNN algorithm for training the network
"""
import os
import matplotlib.pyplot as plt
import tensorflow, numpy as np


"""
Load tHZ Pulse data and return
"""
meas = []
with open('Spektrum_THz.txt') as f:
    meas = f.readlines()[6:]
    x = []
    y = []
    for line in meas:
        if line.strip() != '':
            parts = line.split('\t')
            x.append(float(parts[0]))
            y.append(float(parts[1]))
    #plt.subplot(2,1,1)
    #plt.xlabel('Time in [ps]')
    #plt.ylabel('THz Signal [a.u]')
    #plt.plot(x,y)
    #plt.subplot(2,1,2)
    #x[:] = [i / 33.3333 for i in x]
    #plt.magnitude_spectrum(y,Fs=33.3333)
    #plt.show()
    
    def soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def generate_training_data(num_samples, num_features):
        reflection_coefficients = np.random.randn(num_samples, num_features)
        wavelets = np.random.randn(num_samples, num_features)
        synthetic_seismograms = np.array([np.convolve(rc, w, mode='same') for rc, w in zip(reflection_coefficients, wavelets)])
        return reflection_coefficients, wavelets, synthetic_seismograms
    
    """
    Initialize Parameter of the neural network
    """
    U = np.random.randn()
    W = np.random.randn()
    V = np.tanh # activation function
    iterations = 100
    threshold = 0.1
        
    # Input data
    x = y
    S = np.random.randn(len(x), len(x)) # Matrix S

    """
    Forward computation
    """
    for t in range(iterations):
        x = soft_threshold(np.dot(S, x), threshold)
        x = V(x)
        
    """
    Backward computing
    """
    for t in reversed(range(iterations)):
        error = x - np.dot(S, x)
        x -= U * error  # Update x with error correction

    print("Deconvolved signal:", x)
    plt.subplot(2,1,1)
    plt.xlabel('Time in [ps]')
    plt.ylabel('THz Signal [a.u]')
    plt.plot(x)
    plt.show()
