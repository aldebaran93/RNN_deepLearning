"""
Script for reading the THz pulse and plot it in time domain and in frequency domain as well
"""
import os
import matplotlib.pyplot as plt

with open("Spektrum THz.txt") as f:
    meas = str(f.readlines()[6:]).split("\t")
    for i in meas:
        i.split("\t")
f.close()