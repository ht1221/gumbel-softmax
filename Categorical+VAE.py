
# coding: utf-8

# # Categorical VAE with Gumbel-Softmax
# 
# Partial implementation of the paper [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) 

# A categorical VAE with discrete latent variables. Tensorflow version is 0.10.0.

# # 1. Imports and Helper Functions

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "./data")
from read_data import load_mnist, next_batch
# get_ipython().magic('matplotlib inline')
slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli



# In[2]:


def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y


# # 2. Build Model

# In[3]:


K=10 # number of classes
N=30 # number of categorical distributions


# In[4]:


# input image x (shape=(batch_size,784))
x = tf.placeholder(tf.float32,[None,784])
# variational posterior q(y|x), i.e. the encoder (shape=(batch_size,200))
net = slim.stack(x,slim.fully_connected,[512,256])
# unnormalized logits for N separate K-categorical distributions (shape=(batch_size*N,K))
logits_y = tf.reshape(slim.fully_connected(net,K*N,activation_fn=None),[-1,K])
q_y = tf.nn.softmax(logits_y)
log_q_y = tf.log(q_y+1e-20)
# temperature
tau = tf.Variable(5.0,name="temperature")
# sample and reshape back (shape=(batch_size,N,K))
# set hard=True for ST Gumbel-Softmax
y = tf.reshape(gumbel_softmax(logits_y,tau,hard=False),[-1,N,K])
# generative model p(x|y), i.e. the decoder (shape=(batch_size,200))
net = slim.stack(slim.flatten(y),slim.fully_connected,[256,512])
logits_x = slim.fully_connected(net,784,activation_fn=None)
# (shape=(batch_size,784))
p_x = Bernoulli(logits=logits_x)


# In[5]:


# loss and train ops
kl_tmp = tf.reshape(q_y*(log_q_y-tf.log(1.0/K)),[-1,N,K])
KL = tf.reduce_sum(kl_tmp,[1,2])
elbo=tf.reduce_sum(p_x.log_prob(x),1) - KL


# In[6]:


loss=tf.reduce_mean(-elbo)
lr=tf.constant(0.001)
train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,var_list=slim.get_model_variables())
# init_op=tf.initialize_all_variables()
init_op = tf.global_variables_initializer()

print "Build model successfully!"

# # 3. Train

# In[7]:


# get data
data = load_mnist("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte")
print "Get data successful!"

# In[8]:


BATCH_SIZE=100
NUM_ITERS=50000
tau0=1.0 # initial temperature
np_temp=tau0
np_lr=0.001
ANNEAL_RATE=0.00003
MIN_TEMP=0.5


# In[9]:


dat=[]
sess=tf.InteractiveSession()
sess.run(init_op)
for i in range(1,NUM_ITERS):
  np_x,np_y=next_batch(data, BATCH_SIZE)
  _,np_loss=sess.run([train_op,loss],{
      x:np_x,
      tau:np_temp,
      lr:np_lr
    })
  if i % 100 == 1:
    dat.append([i,np_temp,np_loss])
  if i % 1000 == 1:
    np_temp=np.maximum(tau0*np.exp(-ANNEAL_RATE*i),MIN_TEMP)
    np_lr*=0.9
  if i % 5000 == 1:
    print('Step %d, ELBO: %0.3f' % (i,-np_loss))

'''
# ## save to animation

# In[10]:


np_x1,_=data.next_batch(100)
np_x2,np_y1 = sess.run([p_x.mean(),y],{x:np_x1})


# In[11]:


import matplotlib.animation as animation


# In[12]:


def save_anim(data,figsize,filename):
  fig=plt.figure(figsize=(figsize[1]/10.0,figsize[0]/10.0))
  im = plt.imshow(data[0].reshape(figsize),cmap=plt.cm.gray,interpolation='none')
  plt.gca().set_axis_off()
  #fig.tight_layout()
  fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
  def updatefig(t):
    im.set_array(data[t].reshape(figsize))
    return im,
  anim=animation.FuncAnimation(fig, updatefig, frames=100, interval=50, blit=True, repeat=True)
  Writer = animation.writers['imagemagick']
  writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
  anim.save(filename, writer=writer)
  return


# In[13]:


# save_anim(np_x1,(28,28),'x0.gif')
# save_anim(np_y1,(N,K),'y.gif')
# save_anim(np_x2,(28,28),'x1.gif')


# # 4. Plot Training Curves

# In[14]:


dat=np.array(dat).T


# In[15]:


f,axarr=plt.subplots(1,2)
axarr[0].plot(dat[0],dat[1])
axarr[0].set_ylabel('Temperature')
axarr[1].plot(dat[0],dat[2])
axarr[1].set_ylabel('-ELBO')


# # 5. Unconditional Generation
# 
# This consists of sampling from the prior $p_\theta(y)$ and passing it through the generative model.

# In[16]:


M=100*N
np_y = np.zeros((M,K))
np_y[range(M),np.random.choice(K,M)] = 1
np_y = np.reshape(np_y,[100,N,K])


# In[17]:


x_p=p_x.mean()
np_x= sess.run(x_p,{y:np_y})


# In[18]:


np_y = np_y.reshape((10,10,N,K))
np_y = np.concatenate(np.split(np_y,10,axis=0),axis=3)
np_y = np.concatenate(np.split(np_y,10,axis=1),axis=2)
y_img = np.squeeze(np_y)


# In[19]:


np_x = np_x.reshape((10,10,28,28))
# split into 10 (1,10,28,28) images, concat along columns -> 1,10,28,280
np_x = np.concatenate(np.split(np_x,10,axis=0),axis=3)
# split into 10 (1,1,28,280) images, concat along rows -> 1,1,280,280
np_x = np.concatenate(np.split(np_x,10,axis=1),axis=2)
x_img = np.squeeze(np_x)


# In[26]:


f,axarr=plt.subplots(1,2,figsize=(15,15))
# samples
axarr[0].matshow(y_img,cmap=plt.cm.gray)
axarr[0].set_title('Z Samples')
# reconstruction
axarr[1].imshow(x_img,cmap=plt.cm.gray,interpolation='none')
axarr[1].set_title('Generated Images')


# In[31]:


f.tight_layout()
f.savefig('/Users/ericjang/Desktop/gumbel_softmax/code.png')
'''