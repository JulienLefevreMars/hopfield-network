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
from matplotlib import cm


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
    J = 1/N*epsilon.transpose().dot(epsilon)
    for i in range(N):
        J[i,i] = 0
    return J


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
    return test_array, random_pattern

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
        test_array = np.where(h<0, -1, 1)  # a bit odd
        #     print(test_array.shape)
        for i in range(P):
        #         print(iteration)
            hamming_distance[iteration, i] = ((epsilon - test_array)[i]!=0).sum()
        plt.subplot(N_fig, N_fig,iteration+1)
        plt.imshow(np.where(test_array.reshape(N_sqrt, N_sqrt)<1, 0, 1), cmap=cm.gray)
    return test_array,hamming_distance

def evolution_hopfield_synchronous(test_array,epsilon,w,NO_OF_ITERATIONS,VERBOSE=False):
    # Synchronous version ?
    P = epsilon.shape[0]
    N = epsilon.shape[1]   
    N_sqrt = int(np.sqrt(N))
    hamming_distance = np.zeros((NO_OF_ITERATIONS, P))
    if VERBOSE:
        fig = plt.figure(figsize = (8, 8))
    for iteration in range(NO_OF_ITERATIONS):
        h = w.dot(test_array)
        test_array = np.where(h<0, -1, 1)
        #     print(test_array.shape)
        for i in range(P):
        #         print(iteration)
            hamming_distance[iteration, i] = ((epsilon - test_array)[i]!=0).sum()
        if VERBOSE:
            plt.subplot(5, 5,iteration+1)
            plt.imshow(np.where(test_array.reshape(N_sqrt, N_sqrt)<1, 0, 1), cmap=cm.gray)
    return test_array,hamming_distance

def average_recall(Nstate,epsilon,w,NO_OF_ITERATIONS,P_perturb=0.2):
    P = epsilon.shape[0]
    N = epsilon.shape[1]
    all_recall = np.zeros((Nstate,2))
    for i in tqdm(range(Nstate)):
        perturbed_state, index_state = perturbation(epsilon, p = P_perturb)
        all_recall[i,0] = index_state # index of the random state
        final_state, hamming_distance = evolution_hopfield_synchronous(perturbed_state,epsilon,w,NO_OF_ITERATIONS)
        ind_m1=np.argmin(hamming_distance[-1,:])
        ind_m2=np.argmin(hamming_distance[-2,:])
        if (ind_m1!=ind_m2):
            all_recall[i,1] = -1 # cycle of length 2
        else:
            all_recall[i,1] = ind_m1
    return all_recall
    

def simulation_P_perturb(N,P,sampling,P_sparsity=0.5,NO_OF_STATE=100,VERBOSE=False):
    epsilon_random = np.where(np.random.rand(P,N) < P_sparsity,-1,1)
    w_random = compute_weights_fast(epsilon_random)

    all_P_perturb =np.arange(sampling[0],sampling[1],sampling[2])
    percent_recall = np.zeros((len(all_P_perturb),))
    print(all_P_perturb)
    for i, P_perturb in enumerate(all_P_perturb):
        all_recall = average_recall(NO_OF_STATE,epsilon_random,w_random,NO_OF_ITERATIONS,P_perturb=P_perturb)
        percent_recall[i]=100*np.sum((all_recall[:,0]-all_recall[:,1]) == 0) / NO_OF_STATE
        print(i)
    if VERBOSE:
        plt.plot(all_P_perturb,percent_recall)
    return percent_recall,all_P_perturb

# Load digits images

size = 32
N=size* size

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
plt.figure()
plt.imshow(epsilon[digit,:].reshape(size,size),cmap=cm.gray)

plt.figure(figsize=[6,10])
for digit in range(10):
    plt.subplot(4,3,digit+1)
    plt.imshow(epsilon[digit,:].reshape(size,size),cmap=cm.gray)
    
# Correlation/distance between images
    
distances = np.zeros((10,10,2))
for i in range(10):
    for j in range(10):
        C=np.corrcoef(epsilon[i,:],epsilon[j,:])
        distances[i,j,0]=C[0,1]
        distances[i,j,1]=(epsilon[i,:]!=epsilon[j,:]).sum()
        
plt.imshow(distances[:,:,0])
plt.colorbar()


# Weights for only one image
    
w_1 = compute_weights_fast(epsilon[1:2,:])

plt.figure(figsize=[12,12])
plt.imshow(w_1,cmap=cm.gray)

# Hebian rule to obtain the weights

w = compute_weights_fast(epsilon)
P = epsilon.shape[0]

plt.figure(figsize=[12,12])
plt.imshow(w)
plt.colorbar()

# Associative memory
# Perturbed digit

perturbed_digit, index = perturbation(epsilon,p=0.4,perturbation_type='noise')
plt.imshow(perturbed_digit.reshape(size,size),cmap=cm.gray)

#evolution_hopfield(perturbed_digit,epsilon,w,10)
final_state,hamming_distance=evolution_hopfield_synchronous(perturbed_digit,epsilon,w,10,True)
plt.figure()
plt.plot(hamming_distance)
plt.xlabel('Iterations')
plt.ylabel('Hamming distance')
plt.legend([str(i) for i in range(10)])
"""
From several random configurations, what are the most frequent attractors ?
"""

NO_OF_RANDOM_CONFIG = 500
NO_OF_ITERATIONS = 20
stats = np.zeros((NO_OF_RANDOM_CONFIG,3)) # col 1: pattern where it converges col 2: index of stabilization
p=0.8

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

"""
N and P are constant, increase P_perturb
"""

N=128
P=10
P_sparsity = 0.5 # sparsity level
epsilon_random = np.where(np.random.rand(P,N) < P_sparsity,-1,1)
w_random = compute_weights_fast(epsilon_random)

NO_OF_STATE=100
all_recall = average_recall(NO_OF_STATE,epsilon_random,w_random,NO_OF_ITERATIONS,P_perturb=0.4)
print("\n% of recall = "+ str(100*np.sum((all_recall[:,0]-all_recall[:,1]) == 0) / NO_OF_STATE))

all_P_perturb =np.arange(0.1,0.9,0.02)
percent_recall = np.zeros((len(all_P_perturb),))
for i, P_perturb in enumerate(all_P_perturb):
    all_recall = average_recall(NO_OF_STATE,epsilon_random,w_random,NO_OF_ITERATIONS,P_perturb=P_perturb)
    percent_recall[i]=100*np.sum((all_recall[:,0]-all_recall[:,1]) == 0) / NO_OF_STATE

percent_recall, all_P_perturb=simulation_P_perturb(N,P, [0.2,0.6,0.02],P_sparsity=0.5,NO_OF_STATE=100)
plt.figure()
plt.plot(all_P_perturb,percent_recall)
plt.xlabel('Probability of pixel perturbation')
plt.ylabel('Recall (%)')

"""
N is constant, and increase P/N to see the evolution of the memorization
"""
N=128
if os.path.isfile("All_recall_varying_P_N_128_varying_noise.npz"):
    data=np.load("All_recall_varying_P_N_128_varying_noise.npz")
    all_percent_recall=data['all_percent_recall']
    all_P=data['all_P']
    sample_p=data['sample_p']
else:
    sampling=[0.1,0.7,0.02]
    sample_p=np.arange(sampling[0],sampling[1],sampling[2])
    all_P=np.arange(2,40,4)

    all_percent_recall = np.zeros((len(all_P),len(sample_p)))
    for i, P in enumerate(all_P): 
        percent_recall, all_P_perturb = simulation_P_perturb(N,P,sampling,P_sparsity=0.5,NO_OF_STATE=100)
        all_percent_recall[i,:] = percent_recall
        print("P = "+str(P))
    np.savez("All_recall_varying_P_N_128_varying_noise",all_percent_recall=all_percent_recall,
             all_P=all_P,sample_p=sample_p)

# Image
    
plt.figure()
plt.imshow(all_percent_recall,extent=[0.1,0.7,0.3,0.02])
plt.xlabel('Probability of noise')
plt.ylabel('M/N')
plt.plot([0.1,0.7],[0.14,0.14],'k--')

# Several curves
plt.figure()
plt.plot([],[],' ')
for i in range(len(all_P)):
    plt.plot(sample_p,all_percent_recall[i,:])
plt.legend(["M/N"] + [ '{:04.2f}'.format(i/N) for i in all_P])
plt.xlabel('Probability of pixel perturbation')
plt.ylabel('Recall (%)')

# proba of perturbation fixed =0.1
plt.figure()
plt.plot(all_P,all_percent_recall[:,6])
plt.xlabel("M/N")
plt.ylabel("Recall (%)")

"""
proba fixed = 0.1
"""

if os.path.isfile("All_recall_varying_P_N_128_noise_0.1.npz"):
    data=np.load("All_recall_varying_P_N_128_noise_0.1.npz")
    all_percent_recall_2=data['all_percent_recall_2']
    all_P=data['all_P']
else:
    all_P=np.arange(2,N,2)
    all_percent_recall_2 = np.zeros((len(all_P),1))
    for i, P in enumerate(all_P): 
        percent_recall, all_P_perturb = simulation_P_perturb(N,P,[0.1,0.2,0.2],P_sparsity=0.5,NO_OF_STATE=500)
        all_percent_recall_2[i,:] = percent_recall
        print("P = "+str(P))
    np.savez("All_recall_varying_P_N_128_noise_0.1",all_percent_recall_2=all_percent_recall_2,all_P=all_P)

plt.figure()
plt.plot(all_P,all_percent_recall_2)
plt.semilogx([0.14*N,0.14*N],[0,110],'k--')
plt.xlabel("M")
plt.ylabel("Recall (%)")
plt.text(20,20,"M/N=0.14")
plt.show()

