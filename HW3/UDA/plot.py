import os
import csv
import matplotlib.pyplot as plt
import numpy as np

x_m=[]
y_m=[]
c_m=[]

with open('mnistm.csv', mode='r') as pred:
          reader = csv.reader(pred)
          {x_m.append(float(rows[0])) for rows in reader}

with open('mnistm.csv', mode='r') as pred:
          reader = csv.reader(pred)
          {y_m.append(float(rows[1])) for rows in reader}

with open('mnistm.csv', mode='r') as pred:
          reader = csv.reader(pred)
          {c_m.append(float(rows[2])) for rows in reader}

x_s=[]
y_s=[]
c_s=[]

with open('svhn.csv', mode='r') as pred:
          reader = csv.reader(pred)
          {x_s.append(float(rows[0])) for rows in reader}
          
with open('svhn.csv', mode='r') as pred:
          reader = csv.reader(pred)
          {y_s.append(float(rows[1])) for rows in reader}

with open('svhn.csv', mode='r') as pred:
          reader = csv.reader(pred)
          {c_s.append(float(rows[2])) for rows in reader}
m=np.zeros(len(x_m))
s=np.ones(len(x_s))
x=x_m+x_s
y=y_m+y_s
cl=c_m+c_s
col=np.concatenate((m,s))
# cls_m=np.asarray(c_m)/10
# print(type(x_m[0]) , type(y_m[0]),type(c_m[0]))
# print(x_m.dtype(), y_m.dtype(), c_m.dtype())
fig, ax = plt.subplots(figsize=(30,17))
scatter = ax.scatter(x , y , c= cl , cmap='tab10')
legend1 = ax.legend(*scatter.legend_elements(),
                loc="upper right", title="Classes")
# scatter = ax.scatter((x_s) , (y_s) , c= (c_s) , cmap='tab10', alpha=1, marker='^')

# legend1 = ax.legend(*scatter.legend_elements(),
#                 loc="upper right", title="Classes")
plt.savefig('1.png')
plt.close()
# exit()
print('1 done')
fig, ax = plt.subplots(figsize=(30,17))
scatter = ax.scatter(x , y , c=col,cmap='tab10' )
# scatter1= ax.scatter((x_s) , (y_s) , c='b', alpha=0.3 , marker='^' )
legend1 = ax.legend(*scatter.legend_elements(),
                loc="upper right", title="Domain")


plt.savefig('2.png')
plt.close()
print('done')