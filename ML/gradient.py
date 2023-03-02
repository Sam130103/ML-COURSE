import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv('student_scores.csv')
X=data['Hours'].values
Y=data['Scores'].values

def compute_cost(X,Y,w,b):
    m=X.shape[0]
    total_cost=0
    for i in range(m):
        temp=w*X[i]+b
        total_cost+=(temp*temp)
    return (total_cost/(2**m))
def compute_gradient(X,Y,w,b):
    m=X.shape[0]
    d_w=w
    d_b=b
    for i in range(m):
        temp=w*X[i]+b
        temp1=(temp-Y[i])*X[i]
        temp2=(temp-Y[i])
        d_w+=temp1
        d_b+=temp2
    return (d_w/m),(d_b/m)
def gradient_decent(X,Y,alpha,iter,w,b):
    w_final=w
    b_final=b
    for i in range(iter):
        a,b=compute_gradient(X,Y,w_final,b_final)
        w_final=w_final-alpha*a
        b_final=b_final-alpha*b
    return w_final,b_final
w,b=gradient_decent(X,Y,1e-2,100000,0,0)

plt.scatter(X,Y,c="r",marker="X")
x_train=np.array(X)
y=[]
for i in x_train:
    y.append(w*i+b)
y_train=np.array(y)
plt.plot(x_train,y_train,c="b")
plt.show()


