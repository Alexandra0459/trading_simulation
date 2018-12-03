# In[2]:
from simulation import data_generating as dg
import numpy as np
import math
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def learner1(beta, args):
    (t, K, dg.x, y) = args
    start = t - K - 1
    end = t - 1  # y[t-k-1] ... y[t-2]
    res = y[start:end] - dg.x[start:end].dot(beta)
    return (res.dot(res))


def con(beta):
    return (beta.dot(beta) - 1)


# In[3]:




K = 50
beta1_ori = np.random.randn(dg.T, 3)
beta1_opt = np.zeros((dg.T, 3))
predict_y = np.zeros(dg.T)
predict_f = np.zeros(dg.T)
error1_y = np.zeros(dg.T)
error1_f = np.zeros(dg.T)

VAR1 = np.zeros(dg.T)
VAR2 = np.zeros(dg.T)

cons = [{'type': 'eq', 'fun': con}]

for t in range(K + 1, dg.T):
    para = (t, K, dg.x, dg.y)
    res = opt.minimize(learner1, [0.5, 0.5, 0.5], args=(para,), constraints=cons)

    beta1_opt[t - K] = res.x

    predict_y[t - 1] = dg.y[t - 1] - dg.x[t - 1].dot(res.x)
    predict_f[t - 1] = dg.f[t - 1] - dg.x[t - 1].dot(res.x)

    r_y = dg.y[(t - K - 1):(t - 1)] - dg.x[(t - K - 1):(t - 1)].dot(res.x)
    VAR1[t] = r_y.dot(r_y) / K

    # r_f = f[(t-K):t] - x[(t-K):t].dot(res.x)
    # error1_f[t-K] = r_f.dot(r_f)

    # remaining2 = y[(t-K):t] - x[(t-K):t].dot(beta1_ori[t])
    # error2[t-K] = remaining2.dot(remaining2)

# print(error1)
# print(error2)

VAR1P = predict_y.dot(predict_y) / (dg.T - K - 1)
MSE1P = predict_f.dot(predict_f) / (dg.T - K - 1)
# print(error1_y)
# print(error1_f)
print("VAR1 = ", VAR1P)
print("MSE1 = ", MSE1P)
# print (predict_y)


# In[85]:




error2_y = np.zeros(dg.T)
error2_f = np.zeros(dg.T)
beta2 = np.zeros((dg.T, 10))  # 10: dim of polynomial of power <= 2

predict2_y = np.zeros(dg.T)
predict2_f = np.zeros(dg.T)

for t in range(K + 1, dg.T):
    start = t - K - 1
    end = t - 1
    X_training = dg.x[start:end]
    y_training = dg.y[start:end]
    lin_regressor = LinearRegression()
    poly = PolynomialFeatures(2)
    X_transform = poly.fit_transform(X_training)
    lin_regressor.fit(X_transform, y_training)
    # y_preds = lin_regressor.predict(X_transform) - f_trial
    beta2[t - 1] = lin_regressor.coef_

    X_predict_transform = poly.fit_transform([dg.x[t - 1], ])
    predict2_y[t - 1] = lin_regressor.predict(X_predict_transform) - dg.y[t - 1]
    predict2_f[t - 1] = lin_regressor.predict(X_predict_transform) - dg.f[t - 1]

    r_y = lin_regressor.predict(X_transform) - y_training
    VAR2[t - 1] = r_y.dot(r_y) / K
    # r_f = lin_regressor.predict(X_transform) - f[start:end]
    # error2_f[t-K] = r_f.dot(r_f)

# print(error2_y)
# print(error2_f)

VAR2P = predict2_y.dot(predict2_y) / (dg.T - K - 1)
MSE2P = predict2_f.dot(predict2_f) / (dg.T - K - 1)
print("VAR2 = ", VAR2P)
print("MSE2 = ", MSE2P)

# In[39]:


print(lin_regressor.predict(poly.fit_transform([dg.x[dg.T - 2], ]))[0])

print(beta2[dg.T - 2].dot(poly.fit_transform([dg.x[dg.T - 2], ])[0]))

# In[67]:




LossList3 = np.zeros(dg.T)


def funtionToBeOptimized(a2, b2):
    LossValue = 0

    a1 = 1
    b1 = 1

    for t in range(K + 1 + K1, dg.T):
        S1[t - 1] = a1 * VAR1[t - 2] + a2 * D1[t - 1]
        S2[t - 1] = b1 * VAR2[t - 2] + b2 * D2[t - 1]

    Weight = math.exp(S1[dg.T - 2]) / (math.exp(S1[dg.T - 2]) + math.exp(S2[dg.T - 2]))

    for t in range(K + 1, dg.T):
        Loss = Weight * (dg.x[t].dot(beta1_opt[dg.T - 2])) + (1 - Weight) * (
            poly.fit_transform([dg.x[t], ])[0].dot(beta2[dg.T - 2])) - dg.y[t]
        LossList3[t] = Loss
        LossValue += Loss * Loss

    return LossValue


# In[68]:


K1 = 10
D1 = np.zeros(dg.T)
D2 = np.zeros(dg.T)
S1 = np.zeros(dg.T)
S2 = np.zeros(dg.T)

for t in range(K + 1 + K1, dg.T):
    D1[t - 1] = VAR1[t - 2] - sum(VAR1[(t - K1 - 2): (t - 2)]) / K1
    D2[t - 1] = VAR2[t - 2] - sum(VAR2[(t - K1 - 2): (t - 2)]) / K1

# In[91]:




epochs = 500
params = [1, 1]
learning_rate = 0.0002
batch_size = 10
List_Of_Cost = []


def evaluate_gradient(example, params):
    a2 = params[0]
    b2 = params[1]

    da = (funtionToBeOptimized(a2 + 0.0001, b2) - funtionToBeOptimized(a2, b2)) / 0.0001
    db = (funtionToBeOptimized(a2, b2 + 0.0001) - funtionToBeOptimized(a2, b2)) / 0.0001

    return [da, db]


for i in range(epochs):
    example_list = random.sample(range(K + 1, dg.T), batch_size)
    for example in example_list:
        params_grad = evaluate_gradient(example, params)
        params[0] = params[0] - learning_rate * params_grad[0]
        params[1] = params[1] - learning_rate * params_grad[1]
        List_Of_Cost.append(funtionToBeOptimized(params[0], params[1]) / (dg.T - K - 1))

fig = plt.figure()
plt.plot(np.arange(0, len(List_Of_Cost)), List_Of_Cost)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

# In[84]:


fig = plt.figure()
plt.plot(np.arange(len(List_Of_Cost) - 10, len(List_Of_Cost)),
         List_Of_Cost[len(List_Of_Cost) - 11:len(List_Of_Cost) - 1])
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
print(List_Of_Cost[len(List_Of_Cost) - 6:len(List_Of_Cost) - 1])

# In[78]:


plt.figure()
plt.plot(predict2_y, label='polynomial regression y')
plt.plot(predict_y, label='linear regression of y')
plt.plot(LossList3, label='Combined Models')
plt.xlabel('time')
plt.ylabel('error Y')
plt.legend()
plt.show()

# In[81]:


for t in range(K + 1, dg.T):
    if (abs(LossList3[t]) >= 5):
        print("---------------------")
        print("Loss Combined: ", LossList3[t])
        print("Linear Loss: ", predict_y[t])
        print("Poly Loss: ", predict2_y[t])

    # In[ ]:

for t in range(K + 1, dg.T):
    if (abs(LossList3[t]) >= 5):
        print("---------------------")
        print("Loss Combined: ", LossList3[t])
        print("Linear Loss: ", predict_y[t])
        print("Poly Loss: ", predict2_y[t])

    # In[89]:

print("Variance for Linear regression: ", VAR1P)
print("Variance for Polynomial regression: ", VAR2P)
print("Variance for Combined model: ", List_Of_Cost[-1])
