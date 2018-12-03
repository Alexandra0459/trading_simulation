# In[1]:


import numpy as np
import math

# import tensorflow as tf

# tf.enable_eager_execution()

# tfe = tf.contrib.eager

T = 500
x = np.random.randn(T, 3)
l_margin = 0.1  # probability to shift state
s = np.zeros(T)
y = np.zeros(T)
f = np.zeros(T)

np.random.seed()
l_test = 1

for i in range(T):
    if (l_test < l_margin):
        s[i] = 1 - s[i - 1]
    else:
        s[i] = s[i - 1]

    if (s[i] == 0):
        f[i] = x[i][1] * x[i][1] + x[i][2] * x[i][2] + x[i][0] * x[i][1] + x[i][1] * x[i][2] + x[i][2] * x[i][0]
    else:
        f[i] = x[i][0] + x[i][1]

    epislon = np.random.normal(0, 1)
    y[i] = f[i] + epislon

    l_test = np.random.uniform(0, 1)

# for i in range(T):
# print(y)
# print(f)

