import numpy as np
import matplotlib.pyplot as plt

# https://arxiv.org/abs/1010.4809
spin = 0.77 + 0.068

a = np.genfromtxt("f3_data.txt", delimiter=', ')

plt.plot(a[:,0], a[:,1], 'o')
plt.show()

x = a[:,0] + spin
y = a[:,1]
cov = np.cov(x,y)

print(np.mean(x), np.sqrt(cov[0,0]), np.mean(y), np.sqrt(cov[1,1]), np.sqrt(cov[0,1]*cov[0,1]/(cov[0,0]*cov[1,1])) )
print(cov)


b = np.genfromtxt("f3_data_tex.txt")

x0 = b[:,0]
y0 = b[:,1]

x = -0.2 + (x0-154.560)*(0.1+0.2)/(369.920-154.560) + spin
y = 12 + (y0-665.600)*(20-12)/(880.960-665.600)
cov = np.cov(x,y)

print(np.mean(x), np.sqrt(cov[0,0]), np.mean(y), np.sqrt(cov[1,1]), np.sqrt(cov[0,1]*cov[0,1]/(cov[0,0]*cov[1,1])) )
print(cov)
