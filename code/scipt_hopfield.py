#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:25:03 2020

Script to manipulate hopfield networks

See references:
    http://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11.pdf
    http://ce.sharif.edu/courses/92-93/1/ce957-1/resources/root/Lectures/Lecture23.pdf

@author: julienlefevre
"""

PATH = '../data/digits/'
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import seaborn as sns
sns.set_palette('hls', 10)

size = 32
N=size* size

def image_to_np(path):
    im = Image.open(path)
#     size = 64, 64
#     im.thumbnail(size, Image.ANTIALIAS)
    im_np = np.asarray(im)
    try:
        im_np = im_np[:, :, 0]
    except IndexError:
        pass
    im_np = np.where(im_np<128, -1, 1)
#     plt.imshow(im_np, cmap='gray')
    im_np = im_np.reshape(N)
    return im_np

def compute_weights(epsilon):
    P = epsilon.shape[0]
    N = epsilon.shape[1]
    w = np.zeros((N, N))
    h = np.zeros((N))
    for i in tqdm(range(N)):
        for j in range(N):
            for p in range(P):
                w[i, j] += (epsilon[p, i]*epsilon[p, j])
            if i==j:
                w[i, j] = 0
    w /= N
    return w

def compute_weights_fast(epsilon):
    P = epsilon.shape[0]
    N = epsilon.shape[1]
    return 1/N*(epsilon.transpose().dot(epsilon)-P*np.identity(N))


def perturbation(epsilon,p=0.2,perturbation_type='noise'):
    P = epsilon.shape[0]
    N = epsilon.shape[1]
    if perturbation_type == 'noise':
        random_pattern = np.random.randint(P)
        test_array = epsilon[random_pattern]
        noise = np.random.choice([-1,1],p=[p,1-p], size=N)
        test_array = noise * test_array
    elif perturbation_type == 'mask':
    # test_array = image_to_np(os.path.join(PATH, 'test/test1.jpg'))
        random_pattern = np.random.randint(P)
        test_array = epsilon[random_pattern]
        NO_OF_BITS_TO_CHANGE = int(p*N)
        random_pattern_test = np.random.choice([1, -1], size=NO_OF_BITS_TO_CHANGE)
        test_array[:NO_OF_BITS_TO_CHANGE] = random_pattern_test
    return test_array

def evolution_hopfield(test_array,epsilon,w,NO_OF_ITERATIONS):
    # Asynchronous version ?
    P = epsilon.shape[0]
    N = epsilon.shape[1]   
    N_sqrt = int(np.sqrt(N))
    hamming_distance = np.zeros((NO_OF_ITERATIONS, P))
    fig = plt.figure(figsize = (8, 8))
    N_fig = int(np.sqrt(NO_OF_ITERATIONS))+1
    for iteration in tqdm(range(NO_OF_ITERATIONS)):
        h = np.zeros((N))
        for i in range(N):
            i = np.random.randint(N)
            h[i] = 0
            for j in range(N):
                h[i] += w[i, j]*test_array[j]
        test_array = np.where(h<0, -1, 1)
        #     print(test_array.shape)
        for i in range(P):
        #         print(iteration)
            hamming_distance[iteration, i] = ((epsilon - test_array)[i]!=0).sum()
        plt.subplot(N_fig, N_fig,iteration+1)
        plt.imshow(np.where(test_array.reshape(N_sqrt, N_sqrt)<1, 0, 1), cmap='gray')
    return test_array,hamming_distance

def evolution_hopfield_synchronous(test_array,epsilon,w,NO_OF_ITERATIONS,VERBOSE=False):
    # Synchronous version ?
    P = epsilon.shape[0]
    N = epsilon.shape[1]   
    N_sqrt = int(np.sqrt(N))
    hamming_distance = np.zeros((NO_OF_ITERATIONS, P))
    if VERBOSE:
        fig = plt.figure(figsize = (8, 8))
    for iteration in tqdm(range(NO_OF_ITERATIONS)):
        h = w.dot(test_array)
        test_array = np.where(h<0, -1, 1)
        #     print(test_array.shape)
        for i in range(P):
        #         print(iteration)
            hamming_distance[iteration, i] = ((epsilon - test_array)[i]!=0).sum()
        if VERBOSE:
            plt.subplot(5, 5,iteration+1)
            plt.imshow(np.where(test_array.reshape(N_sqrt, N_sqrt)<1, 0, 1), cmap='gray')
    return test_array,hamming_distance


    

# Load digits images

epsilon = np.asarray([image_to_np(os.path.join(PATH, '0.jpg')),
                     image_to_np(os.path.join(PATH, '1.jpg')),
                     image_to_np(os.path.join(PATH, '2.jpg')),
                     image_to_np(os.path.join(PATH, '3.jpg')),
                     image_to_np(os.path.join(PATH, '4.jpg')),
                     image_to_np(os.path.join(PATH, '5.jpg')),
                     image_to_np(os.path.join(PATH, '6.jpg')),
                     image_to_np(os.path.join(PATH, '7.jpg')),
                     image_to_np(os.path.join(PATH, '8.jpg')),
                     image_to_np(os.path.join(PATH, '9.jpg'))])

   
digit = 7 
plt.imshow(epsilon[digit,:].reshape(size,size))

# Hebian rule to obtain the weights

w = compute_weights_fast(epsilon)
P = epsilon.shape[0]

# Associative memory
# Perturbed digit

perturbed_digit = perturbation(epsilon,p=0.3,perturbation_type='noise')
plt.imshow(perturbed_digit.reshape(size,size))

#evolution_hopfield(perturbed_digit,epsilon,w,10)
final_state,hamming_distance=evolution_hopfield_synchronous(perturbed_digit,epsilon,w,10,True)

"""
From several random configurations, what are the most frequent attractors ?
"""

NO_OF_RANDOM_CONFIG = 500
NO_OF_ITERATIONS = 20
stats = np.zeros((NO_OF_RANDOM_CONFIG,3)) # col 1: pattern where it converges col 2: index of stabilization
p=0.1

for n in range(NO_OF_RANDOM_CONFIG):
    random_state = np.random.choice([-1,1],p=[p,1-p], size=N)
    final_state,hamming_distance=evolution_hopfield_synchronous(random_state,epsilon,w,NO_OF_ITERATIONS)
    stats[n,0] = np.argmin(hamming_distance[-1,:])
    indices = np.where(np.diff(hamming_distance[:,int(stats[n,0])]) ==0)
    if indices[0].size==0:
        stats[n,1] = -1
        stats[n,2] = False
    else:
        stats[n,1] = indices[0][0]
        stats[n,2] = len(indices[0]) == (indices[0][-1]-indices[0][0])+1

plt.figure()
plt.subplot(1,2,1)
plt.hist(stats[:,0],range(0,P+1))
plt.xlabel('Final state')
plt.subplot(1,2,2)
plt.hist(stats[:,1])
plt.xlabel('Index where the iterations stabilize')