import numpy as np  
import matplotlib.pyplot as plt  
import random  
x=[1.50,2.00,2.50,3.00,3.50,4.00,6.00]  
y=[64.50,74.50,84.50,94.50,114.50,154.50,184.50]  
test_x=[1.55,2.00,3.00,4.00,5.00,6.00]  
###随机设置一下参数  
theta0=0.3  
theta1=0.3  

###学习率  
a=0.012  
m=len(x)  

def h_(x):  
    return theta0+theta1*x   

def h(i):  
    return theta0+theta1*x[i]  

def diff(i):  
    return h(i)-y[i]    
###-------------------------------随机梯度  
x = np.array(x).T  
y = np.array(y).T  
plt.plot(x, y, 'r.')  
for times in range(100):  
    sum1=0  
    sum2=0  
    i = random.randint(0,2)  #每次迭代在input_x中随机选取一组样本进行权重的更新  
    ###更新公式  
    sum1=sum1+diff(i)  
    sum2=sum2+diff(i)*x[i]  
    theta0=theta0-(a/m)*sum1     
    theta1=theta1-(a/m)*sum2  
   ### plt.plot(test_x,  [h_(xi)  for xi in test_x ])#把每次迭代的制图  
plt.plot(test_x,  [h_(xi)  for xi in test_x ],color="g", linestyle="-", linewidth=1, label="BGD")    
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))      

plt.title("Figure 1")    
print ("BGD")    
print ("theta0 : ",theta0)    
print ("theta1 : ",theta1)    
###---------------------------------批量梯度    
for times in range(100):  
    sum1=0  
    sum2=0  
    for i in range(m):    
   
   ####更新公式  
      sum1=sum1+diff(i)  
      sum2=sum2+diff(i)*x[i]  
      theta0=theta0-(a/m)*sum1     
      theta1=theta1-(a/m)*sum2  
plt.plot(test_x,  [h_(xi)  for xi in test_x ],color="b", linestyle="-", linewidth=1,label="SGD")  
plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))  
plt.title("Figure 1")  
print ("SGD")  
print ("theta0 : ",theta0)  
print ("theta1 : ",theta1)  
plt.show()  
###把每次迭代的制图  
x=[1.50,2.00,2.50,3.00,3.50,4.00,6.00]  
y=[64.50,74.50,84.50,94.50,114.50,154.50,184.50]  
test_x=[1.55,2.00,3.00,4.00,5.00,6.00]    

###随机设置一下参数  
theta0=0.3  
theta1=0.3  
###学习率  
a=0.012  
m=len(x)  
def h_(x):  
    return theta0+theta1*x   
def h(i):  
    return theta0+theta1*x[i]  
def diff(i):  
    return h(i)-y[i]  
for times in range(100):  
    sum1=0  
    sum2=0  
    for i in range(m):  
        sum1=sum1+diff(i)  
        sum2=sum2+diff(i)*x[i]  
        theta0=theta0-(a/m)*sum1  
        theta1=theta1-(a/m)*sum2  
        plt.plot(test_x,  [h_(xi)  for xi in test_x ])  
###最终结果标注一下  
plt.plot(test_x,  [h_(xi)  for xi in test_x ],'b+')  
print ("theta0 : ",theta0)  
print ("theta1 : ",theta1)  
plt.show()    

BGD  
theta0 :  8.777373926033802  
theta1 :  16.910683341593195  
SGD  
theta0 :  15.34014074719618  
theta1 :  29.348745394676136    
 
theta0 :  14.0802581104996  
theta1 :  29.771119853144558    

 
import numpy as np  
import matplotlib.pyplot as plt  
import random  

x=[1.1,1.3,1.5,2,2.2,2.9,3,3.2,3.2,3.7,3.9,4,4,4.1,4.5,4.9,5.1,5.3,5.9,6,6.8,7.1,7.9,8.2,8.7,9,9.5,9.6,10.3,10.5]  
y=[39343,46205,37731,43525,39891,56642,60150,54445,64445,57189,63218,55794,56957,57081,61111,67938,66029,83088,81363,93940,91738,98273,101302,113812,109431,105582,116969,112635,122391,121872]   

test_x=[1,2,3,4,5,6,7,8,9,10,11,12,12.5,13,14,15]  

###参数，这个是随机的  
theta0=0.1  
theta1=0.2  

###学习率  
alpha=0.01  
m=len(x)    

def h_(x):  
    return theta0+theta1*x   
  
def h(i):  
    return theta0+theta1*x[i]  

def diff(i):  
    return h(i)-y[i]  

for times in range(5000):  
        sum1=0  
        sum2=0  
        i = random.randint(0,3)  
        sum1=sum1+diff(i)  
        sum2=sum2+diff(i)*x[i]  
        theta0=theta0-(alpha/m)*sum1  
        theta1=theta1-(alpha/m)*sum2  
        plt.plot(test_x,  [h_(xi)  for xi in test_x ])  
  
plt.plot(test_x,  [h_(xi)  for xi in test_x ],'b')  
print ("theta0 : ",theta0)  
print ("theta1 : ",theta1)  
plt.show()    

theta0 :  13953.788293719332  
theta1 :  18113.60117477492  

 


###-------------------------------随机梯度  
x = np.array(x).T  
y = np.array(y).T  
plt.plot(x, y, 'r.')  
for times in range(5000):  
    sum1=0  
    sum2=0  
    i = random.randint(0,6)  #每次迭代在input_x中随机选取一组样本进行权重的更新  
    #更新公式  
    sum1=sum1+diff(i)  
    sum2=sum2+diff(i)*x[i]  
    theta0=theta0-(a/m)*sum1     
    theta1=theta1-(a/m)*sum2  
   ### plt.plot(test_x,  [h_(xi)  for xi in test_x ])#把每次迭代的制图  
plt.plot(test_x,  [h_(xi)  for xi in test_x ],color="g", linestyle="-", linewidth=1, label="BGD")  
plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))  

plt.title("Figure 1")  
print ("BGD")  
print ("theta0 : ",theta0)  
print ("theta1 : ",theta1)  
###---------------------------------批量梯度  
for times in range(5000):  
    sum1=0  
    sum2=0  
    for i in range(m):    
   
   ###更新公式  
      sum1=sum1+diff(i)  
      sum2=sum2+diff(i)*x[i]  
      theta0=theta0-(a/m)*sum1     
      theta1=theta1-(a/m)*sum2  
plt.plot(test_x,  [h_(xi)  for xi in test_x ],color="b", linestyle="-", linewidth=1,label="SGD")  
plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))  
plt.title("Figure 1")  
print ("SGD")  
print ("theta0 : ",theta0)  
print ("theta1 : ",theta1)  
plt.show()  

BGD  
theta0 :  11338.726631154344  
theta1 :  16690.987771097793  
SGD  
theta0 :  25585.86790581674  
theta1 :  9582.334792901282  
 

