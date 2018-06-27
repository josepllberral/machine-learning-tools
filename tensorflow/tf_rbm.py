################################################################################
# RESTRICTED BOLTZMANN MACHINES in TensorFlow                                  #
################################################################################ 

## @authors Josep Ll. Berral (Barcelona Supercomputing Center)
 
## @date 27th June, 2018
 
## @license GNU GPL v3 (or posterior) http://www.gnu.org/licenses/

## Here you can find a Restricted Boltzmann Machines implementation in TF, for
## academic and educational purposes.

## References
## * Approach based on XXX RBM: TODO

from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys

################################################################################
# GENERIC FUNCTIONS                                                            #
################################################################################
 
## * Xavier Glorot's initialization method
## * Sampling Bernoulli
## * Sampling Gaussian

def glorot_init(fan_in, fan_out, const = 1.0, dtype = np.float32):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval = -k, maxval = k, dtype = dtype)

def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

def sample_gaussian(x, sigma):
    return x + tf.random_normal(tf.shape(x), mean = 0.0, stddev = sigma, dtype = tf.float32)

################################################################################
# RBM FUNCTIONS                                                                #
################################################################################
 
## Generic multi-purpose RBM
 
## Constructor:
## * creates the RBM structure and TF execution tree. Defines the number of
##   visible units (inputs) and number of hidden units, also the hyperparameters

## Training:
## * fit: trains the RBM using the provided inputs
## * evaluate_error: evaluates the reconstruction error of passing data forward
##   and backward through the RBM

## Predicting/Reconstructing:
## * forward: propagates the inputs through the RBM towards the hidden layer
##   activations
## * backward: propagates the activations through the RBM backwards to the
##   visible layer input reconstructions
## * forward_backward: propagates the inputs through the RBM towards the hidden
##   layer activations, then the activations backwards to the visible layer
##   input reconstructions
## * gibbs_sampling: reconstructs inputs using Gibbs sampling passing inputs
##   forward and backward the RBM

## Setters and Getters:
## * get_weights: gets the weights of the RBM for an external source
## * set_weights: sets the weights of the RBM from an external source
## * save_weights: saves the weights of the RBM into a file
## * load_weights: loads the weights of the RBM from a file

## Gaussian-Bernoulli RBM
class RBM:
    
    # Constructor of RBM, define execution plan and launch TF
    def __init__(self, n_visible, n_hidden, learning_rate = 0.01, momentum = 0.8):
        assert momentum >= 0.0 and momentum < 1
        assert learning_rate >= 0.0 and learning_rate < 1
        
        '''
        Initialize RBM variables
        '''
        self.n_visible     = n_visible
        self.n_hidden      = n_hidden
        self.learning_rate = learning_rate
        self.momentum      = momentum
        
        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])
        
        self.w     = tf.Variable(glorot_init(self.n_visible, self.n_hidden), dtype = tf.float32)
        self.vbias = tf.Variable(tf.zeros([self.n_visible]), dtype = tf.float32)
        self.hbias = tf.Variable(tf.zeros([self.n_hidden]), dtype = tf.float32)
        
        self.delta_w     = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype = tf.float32)
        self.delta_vbias = tf.Variable(tf.zeros([self.n_visible]), dtype = tf.float32)
        self.delta_hbias = tf.Variable(tf.zeros([self.n_hidden]), dtype = tf.float32)
        
        '''
        TF execution plan for Contrastive Divergence-k
        '''
        
        # compute positive phase
        ph_mean = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hbias)
        ph_sample = sample_bernoulli(ph_mean)
        
        # compute negative phase (k = 1)
        nv = tf.matmul(ph_sample, tf.transpose(self.w)) + self.vbias
        nh_mean = tf.nn.sigmoid(tf.matmul(nv, self.w) + self.hbias)
        nh_sample = sample_bernoulli(nh_mean)
        
        # determine gradients on RBM parameters
        gradient_w = tf.matmul(tf.transpose(self.x), ph_mean) - tf.matmul(tf.transpose(nv), nh_mean)
        gradient_v = tf.reduce_mean(self.x - nv, 0)
        gradient_h = tf.reduce_mean(ph_mean - nh_mean, 0)
        
        new_delta_w     = self.momentum * self.delta_w     + (1 - self.momentum) * self.learning_rate * gradient_w / tf.to_float(tf.shape(gradient_w)[0])
        new_delta_vbias = self.momentum * self.delta_vbias + (1 - self.momentum) * self.learning_rate * gradient_v / tf.to_float(tf.shape(gradient_v)[0])
        new_delta_hbias = self.momentum * self.delta_hbias + (1 - self.momentum) * self.learning_rate * gradient_h / tf.to_float(tf.shape(gradient_h)[0])
        
        # update weights and deltas
        update_delta_w     = self.delta_w.assign(new_delta_w)
        update_delta_vbias = self.delta_vbias.assign(new_delta_vbias)
        update_delta_hbias = self.delta_hbias.assign(new_delta_hbias)
        
        update_w     = self.w.assign(self.w + new_delta_w)
        update_vbias = self.vbias.assign(self.vbias + new_delta_vbias)
        update_hbias = self.hbias.assign(self.hbias + new_delta_hbias)
        
        self.update_deltas  = [update_delta_w, update_delta_vbias, update_delta_hbias]
        self.update_weights = [update_w, update_vbias, update_hbias]
        
        '''
        TF execution plan for passing data forward and backward
        '''
        # compute values forward and backward
        self.activation_mean   = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hbias)
        self.activation_sample = sample_bernoulli(self.activation_mean)
        self.reconstruction    = tf.matmul(self.activation_sample, tf.transpose(self.w)) + self.vbias
        self.decodification    = tf.matmul(self.y, tf.transpose(self.w)) + self.vbias
        
        # approximation to the reconstruction error
        self.compute_err = tf.reduce_mean(tf.square(self.x - self.reconstruction))
        
        '''
        Initialize TF Session
        '''
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    
    # Forward and Backward functions
    def forward(self, batch_x):
        return self.sess.run(self.activation_sample, feed_dict = {self.x: batch_x})
    
    def backward(self, batch_y):
        return self.sess.run(self.decodification, feed_dict = {self.y: batch_y})
    
    def forward_backward(self, batch_x):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: batch_x})
    
    def evaluate_error(self, batch_x):
        return self.sess.run(self.compute_err, feed_dict = {self.x: batch_x})
    
    # Gibbs sampling for data generation
    def gibbs_sampling(self, batch_x, n_gibbs = 30):
        # Positive phase
        ph_mean = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.hbias)
        ph_sample = sample_bernoulli(ph_mean)
        
        # Negative phase
        nh_sample = ph_sample
        for i in range(n_gibbs):
            nv = tf.matmul(nh_sample, tf.transpose(self.w)) + self.vbias
            nh_mean = tf.nn.sigmoid(tf.matmul(nv, self.w) + self.hbias)
            nh_sample = sample_bernoulli(nh_mean)
        self.last_reconstructed = nv
        
        # TF compute
        return self.sess.run(self.last_reconstructed, feed_dict = {self.x: batch_x})
    
    # How to train your RBM
    def fit(self, data_x, n_epochs = 10, batch_size = 10, verbose = True):
        assert n_epochs > 0
        
        n_data = data_x.shape[0]
        
        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1
        
        data_x_cpy = data_x.copy()
        inds = np.arange(n_data)
        
        errs = []
        
        for e in range(n_epochs):
            
            np.random.shuffle(inds)
            data_x_cpy = data_x_cpy[inds]
            
            r_batches = range(n_batches)
            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0
            
            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                self.sess.run(self.update_weights + self.update_deltas, feed_dict = {self.x: batch_x})
                epoch_errs[epoch_errs_ptr] = self.sess.run(self.compute_err, feed_dict = {self.x: batch_x})
                epoch_errs_ptr += 1
            
            if verbose:
                print('Epoch: {:d}'.format(e), 'Train error: {:.4f}'.format(epoch_errs.mean()))
                sys.stdout.flush()
            
            errs = np.hstack([errs, epoch_errs])
        
        return errs
    
    # Additional functions
    def get_weights(self):
        return self.sess.run(self.w), self.sess.run(self.vbias), self.sess.run(self.hbias)
    
    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w, name + '_v': self.vbias, name + '_h': self.hbias})
        return saver.save(self.sess, filename)
    
    def set_weights(self, w, vbias, hbias):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.vbias.assign(vbias))
        self.sess.run(self.hbias.assign(hbias))
    
    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w, name + '_v': self.vbias, name + '_h': self.hbias})
        saver.restore(self.sess, filename)
