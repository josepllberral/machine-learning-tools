################################################################################
# CONDITIONAL RESTRICTED BOLTZMANN MACHINES in TensorFlow                      #
################################################################################

## @authors Josep Ll. Berral (Barcelona Supercomputing Center)

## @date 27th June, 2018

## @license GNU GPL v3 (or posterior) http://www.gnu.org/licenses/

## Here you can find a Conditional Restricted Boltzmann Machines, for academic
## and educational purposes. Two versions are presented:
## * CRBM with explicit conditioner
## * Extension of CRBM considering inputs as Time-Series (conditioner="history")

## References
## * Approach based on Graham Taylor's CRBM: http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/gwtaylor_nips.pdf

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
# CRBM FUNCTIONS                                                               #
################################################################################

## Generic multi-purpose CRBM

## Constructor:
## * creates the CRBM structure and TF execution tree. Defines the number of
##   visible units (inputs), number of condition units (static inputs) and
##   number of hidden units, also the hyperparameters.

## Training:
## * fit: trains the CRBM using the provided inputs and condition inputs
## * evaluate_error: evaluates the reconstruction error of passing data forward
##   and backward through the CRBM

## Predicting/Reconstructing:
## * forward: propagates the inputs and condition inputs through the CRBM
##   towards the hidden layer activations
## * backward: propagates the activations and condition inputs through the CRBM
##   backwards to the visible layer input reconstructions
## * forward_backward: propagates the inputs and condition inputs through the
##   CRBM towards the hidden layer activations, then the activations and
##   condition inputs backwards to the visible layer input reconstructions
## * gibbs_sampling: reconstructs inputs using Gibbs sampling passing inputs
##   forward and backward the CRBM

## Setters and Getters:
## * **get_weights**: gets the weights of the CRBM for an external source
## * **set_weights**: sets the weights of the CRBM from an external source
## * **save_weights**: saves the weights of the CRBM into a file
## * **load_weights**: loads the weights of the CRBM from a file

## Gaussian-Bernoulli Conditional-RBM
class CRBM:
    
    # Constructor of CRBM, define execution plan and launch TF
    def __init__(self, n_visible, n_condition, n_hidden, learning_rate = 0.01, momentum = 0.8):
        assert momentum >= 0.0 and momentum <= 1
        assert learning_rate >= 0.0 and learning_rate < 1
        
        '''
        Initialize CRBM variables
        '''
        self.n_visible     = n_visible
        self.n_hidden      = n_hidden
        self.n_condition   = n_condition
        self.learning_rate = learning_rate
        self.momentum      = momentum
        
        self.x = tf.placeholder(tf.float32, [None, self.n_visible])
        self.y = tf.placeholder(tf.float32, [None, self.n_hidden])
        self.c = tf.placeholder(tf.float32, [None, self.n_condition])
        
        self.w     = tf.Variable(glorot_init(self.n_visible, self.n_hidden), dtype = tf.float32)
        self.vbias = tf.Variable(tf.zeros([self.n_visible]), dtype = tf.float32)
        self.hbias = tf.Variable(tf.zeros([self.n_hidden]), dtype = tf.float32)
        
        self.a     = tf.Variable(glorot_init(self.n_condition, self.n_visible), dtype = tf.float32)
        self.b     = tf.Variable(glorot_init(self.n_condition, self.n_hidden), dtype = tf.float32)
        
        self.delta_w     = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype = tf.float32)
        self.delta_vbias = tf.Variable(tf.zeros([self.n_visible]), dtype = tf.float32)
        self.delta_hbias = tf.Variable(tf.zeros([self.n_hidden]), dtype = tf.float32)
        
        self.delta_a     = tf.Variable(tf.zeros([self.n_condition, self.n_visible]), dtype = tf.float32)
        self.delta_b     = tf.Variable(tf.zeros([self.n_condition, self.n_hidden]), dtype = tf.float32)
        
        '''
        TF execution plan for Contrastive Divergence-k
        '''
        # compute positive phase
        ph_mean = tf.nn.sigmoid(tf.matmul(self.x, self.w) + tf.matmul(self.c, self.b) + self.hbias)
        ph_sample = sample_bernoulli(ph_mean)
        
        # compute negative phase (k = 1)
        nv = tf.matmul(ph_sample, tf.transpose(self.w)) + tf.matmul(self.c, self.a) + self.vbias
        nh_mean = tf.nn.sigmoid(tf.matmul(nv, self.w) + tf.matmul(self.c, self.b) + self.hbias)
        nh_sample = sample_bernoulli(nh_mean)
        
        # determine gradients on CRBM parameters
        gradient_w = tf.matmul(tf.transpose(self.x), ph_mean) - tf.matmul(tf.transpose(nv), nh_mean)
        gradient_v = tf.reduce_mean(self.x - nv, 0)
        gradient_h = tf.reduce_mean(ph_mean - nh_mean, 0)
        gradient_a = tf.matmul(tf.transpose(self.c), self.x) - tf.matmul(tf.transpose(self.c), nv)
        gradient_b = tf.matmul(tf.transpose(self.c), ph_mean) - tf.matmul(tf.transpose(self.c), nh_mean)
        
        new_delta_w     = self.momentum * self.delta_w     + (1 - self.momentum) * self.learning_rate * gradient_w / tf.to_float(tf.shape(gradient_w)[0])
        new_delta_vbias = self.momentum * self.delta_vbias + (1 - self.momentum) * self.learning_rate * gradient_v / tf.to_float(tf.shape(gradient_v)[0])
        new_delta_hbias = self.momentum * self.delta_hbias + (1 - self.momentum) * self.learning_rate * gradient_h / tf.to_float(tf.shape(gradient_h)[0])
        new_delta_a     = self.momentum * self.delta_a     + (1 - self.momentum) * self.learning_rate * gradient_a / tf.to_float(tf.shape(gradient_a)[0])
        new_delta_b     = self.momentum * self.delta_b     + (1 - self.momentum) * self.learning_rate * gradient_b / tf.to_float(tf.shape(gradient_b)[0])
        
        # update weights and deltas
        update_delta_w     = self.delta_w.assign(new_delta_w)
        update_delta_vbias = self.delta_vbias.assign(new_delta_vbias)
        update_delta_hbias = self.delta_hbias.assign(new_delta_hbias)
        update_delta_a     = self.delta_a.assign(new_delta_a)
        update_delta_b     = self.delta_b.assign(new_delta_b)
        
        update_w     = self.w.assign(self.w + new_delta_w)
        update_vbias = self.vbias.assign(self.vbias + new_delta_vbias)
        update_hbias = self.hbias.assign(self.hbias + new_delta_hbias)
        update_a     = self.a.assign(self.a + new_delta_a)
        update_b     = self.b.assign(self.b + new_delta_b)
        
        self.update_deltas  = [update_delta_w, update_delta_vbias, update_delta_hbias, update_delta_a, update_delta_b]
        self.update_weights = [update_w, update_vbias, update_hbias, update_a, update_b]
        
        '''
        TF execution plan for passing data forward and backward
        '''
        # compute values forward and backward
        self.activation_mean   = tf.nn.sigmoid(tf.matmul(self.x, self.w) + tf.matmul(self.c, self.b) + self.hbias)
        self.activation_sample = sample_bernoulli(self.activation_mean)
        self.reconstruction    = tf.matmul(self.activation_sample, tf.transpose(self.w)) + tf.matmul(self.c, self.a) + self.vbias
        self.decodification    = tf.matmul(self.y, tf.transpose(self.w)) + tf.matmul(self.c, self.a) + self.vbias
        
        # approximation to the reconstruction error
        self.compute_err = tf.reduce_mean(tf.square(self.x - self.reconstruction))
        
        '''
        Initialize TF Session
        '''
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    
    # Forward and Backward functions
    def forward(self, data_x, data_c):
        assert data_x.shape[0] == data_c.shape[0]
        return self.sess.run(self.activation_sample, feed_dict = {self.x: data_x, self.c: data_c})
    
    def backward(self, data_y, data_c):
        assert data_y.shape[0] == data_c.shape[0]
        return self.sess.run(self.decodification, feed_dict = {self.y: data_y, self.c: data_c})
    
    def forward_backward(self, data_x, data_c):
        assert data_x.shape[0] == data_c.shape[0]
        return self.sess.run(self.reconstruction, feed_dict = {self.x: data_x, self.c: data_c})
    
    def evaluate_error(self, data_x, data_c):
        assert data_x.shape[0] == data_c.shape[0]
        return self.sess.run(self.compute_err, feed_dict = {self.x: data_x, self.c: data_c})
    
    # Gibbs sampling for data generation
    def gibbs_sampling(self, data_x, data_c, n_gibbs = 30):
        assert data_x.shape[0] == data_c.shape[0]
        assert n_gibbs > 0
        
        # Positive phase
        ph_mean = tf.nn.sigmoid(tf.matmul(self.x, self.w) + tf.matmul(self.c, self.b) + self.hbias)
        ph_sample = sample_bernoulli(ph_mean)
        
        # Negative phase
        nh_sample = ph_sample
        for i in range(n_gibbs):
            nv = tf.matmul(nh_sample, tf.transpose(self.w)) + tf.matmul(self.c, self.a) + self.vbias
            nh_mean = tf.nn.sigmoid(tf.matmul(nv, self.w) + tf.matmul(self.c, self.b) + self.hbias)
            nh_sample = sample_bernoulli(nh_mean)
        self.last_reconstructed = nv
        
        return self.sess.run(self.last_reconstructed, feed_dict = {self.x: data_x, self.c: data_c})
    
    # How to train your CRBM
    def fit(self, data_x, data_c, n_epochs = 10, batch_size = 10, verbose = True):
        assert n_epochs > 0
        assert data_x.shape[0] == data_c.shape[0]
        assert self.n_visible == data_x.shape[1] and self.n_visible * self.delay == data_c.shape[1]
        
        n_data = data_x.shape[0]
        
        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1
        
        data_x_cpy = data_x.copy()
        data_c_cpy = data_c.copy()
        inds = np.arange(n_data)
        
        errs = []
        
        for e in range(n_epochs):
            
            np.random.shuffle(inds)
            data_x_cpy = data_x_cpy[inds]
            data_c_cpy = data_c_cpy[inds]
            
            r_batches = range(n_batches)
            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0
            
            for b in r_batches:
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                batch_c = data_c_cpy[b * batch_size:(b + 1) * batch_size]
                self.sess.run(self.update_weights + self.update_deltas, feed_dict = {self.x: batch_x, self.c: batch_c})
                epoch_errs[epoch_errs_ptr] = self.sess.run(self.compute_err, feed_dict = {self.x: batch_x, self.c: batch_c})
                epoch_errs_ptr += 1
            
            if verbose:
                print('Epoch: {:d}'.format(e), 'Train error: {:.4f}'.format(epoch_errs.mean()))
                sys.stdout.flush()
            
            errs = np.hstack([errs, epoch_errs])
        
        return errs
    
    # Additional functions
    def get_weights(self):
        return self.sess.run(self.w), self.sess.run(self.vbias), self.sess.run(self.hbias), self.sess.run(self.a), self.sess.run(self.b)
    
    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w, name + '_v': self.vbias, name + '_h': self.hbias, name + '_a': self.a, name + '_b': self.b})
        return saver.save(self.sess, filename)
    
    def set_weights(self, w, vbias, hbias, a, b):
        self.sess.run(self.w.assign(w))
        self.sess.run(self.vbias.assign(vbias))
        self.sess.run(self.hbias.assign(hbias))
        self.sess.run(self.a.assign(a))
        self.sess.run(self.b.assign(b))
    
    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.w, name + '_v': self.vbias, name + '_h': self.hbias, name + '_a': self.a, name + '_b': self.b})
        saver.restore(self.sess, filename)

## Time-series CRBM
 
## Extra functions of the TS-CRBM:
## * fit: inherits from the CRBM. Condition inputs are automatically generated
##   as the delay-history for each input. The input data can be composed by
##   different series, so series lenghts should be introduced.
## * build-history: internal function that generates the conditioning history
##   data for each point. This function is called by the other functions.
## * forecast: given an input data point and its history, generates n following
##   steps using Gibbs sampling using the CRBM

## Gaussian-Bernoulli Conditional-RBM for Time-Series
class CRBM_Series(CRBM):
    # Override Constructor
    def __init__(self, n_visible, n_hidden, delay = 6, learning_rate = 0.01, momentum = 0.8):
        assert delay > 0
        
        self.delay = delay       
        CRBM.__init__(self, n_visible, n_visible * delay, n_hidden, learning_rate, momentum)
    
    def build_history(self, data_x, seqlen = None):
        if seqlen is None:
            seqlen = [data_x.shape[0]]
        data_x_new = np.empty((0,self.n_visible))
        data_c = np.empty((0, self.n_visible * self.delay))
        idx = 0
        for s in seqlen:
            if s > self.delay:
                data_xs = data_x[idx:(idx + s),:]
                data_cs = np.zeros((data_xs.shape[0] - self.delay, self.n_visible * self.delay))
                for i in range(self.delay, data_xs.shape[0]):
                    data_cs[(i - self.delay),:] = data_xs[(i - self.delay):i,:].reshape((self.n_visible * self.delay))
                data_xs = np.matrix(data_xs[self.delay:,:],)
                data_x_new = np.append(data_x_new, data_xs, axis = 0)
                data_c = np.append(data_c, data_cs, axis = 0)
            idx = idx + s
        return data_x_new, data_c
    
    # Override Forward and Backward functions
    def forward(self, data_x, seqlen = None, data_c = None):
        if data_c is None:
            data_x, data_c = self.build_history(data_x, seqlen)
        return super(CRBM_Series, self).forward(data_x, data_c)
    
    def forward_backward(self, data_x, seqlen = None, data_c = None):
        if data_c is None:
            data_x, data_c = self.build_history(data_x, seqlen)
        return super(CRBM_Series, self).forward_backward(data_x, data_c)
    
    def evaluate_error(self, data_x, seqlen = None, data_c = None):
        if data_c is None:
            data_x, data_c = self.build_history(data_x, seqlen)
        return super(CRBM_Series, self).evaluate_error(data_x, data_c)
    
    # Override Gibbs sampling for data generation
    def gibbs_sampling(self, data_x, seqlen = None, data_c = None, n_gibbs = 30):
        if data_c is None:
            data_x, data_c = self.build_history(data_x, seqlen)
        return super(CRBM_Series, self).gibbs_sampling(data_x, data_c, n_gibbs)
    
    # Forecast n steps using Gibbs sampling for data generation
    def forecast(self, data_x, n_samples = 10, n_gibbs = 30):
        assert n_gibbs > 0 and n_samples > 0
        
        vis = tf.identity(self.x)
        for h in range(n_gibbs):
            hid_mean = tf.nn.sigmoid(tf.matmul(vis, self.w) + tf.matmul(self.c, self.b) + self.hbias)
            hid_sample = sample_bernoulli(hid_mean)
            vis = tf.matmul(hid_sample, tf.transpose(self.w)) + tf.matmul(self.c, self.a) + self.vbias
            vis = tf.clip_by_value(vis, -2, 2)
        self.generate_op = vis
        
        sequence = np.zeros([n_samples, self.n_visible], dtype = float)
        
        persist_window = data_x[-1 - self.delay:,:].reshape((self.delay + 1) * self.n_visible)
        
        for t in range(n_samples):
            sequence[t,:] = self.sess.run(self.generate_op,
                                          feed_dict = {
                                              self.x: np.matrix((persist_window[:self.n_visible])),
                                              self.c: np.matrix((persist_window[self.n_visible:]))
                                          })
            persist_window = np.append(persist_window[self.n_visible:], sequence[t,:].copy())
        
        return sequence
    
    # How to train your CRBM_Series
    def fit(self, data_x, seqlen = None, data_c = None, n_epochs = 10, batch_size = 10, verbose = True):
        if data_c is None:
            data_x, data_c = self.build_history(data_x, seqlen)
        return super(CRBM_Series, self).fit(data_x, data_c, n_epochs, batch_size, verbose)
