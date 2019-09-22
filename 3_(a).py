import numpy as np  
import matplotlib.pyplot as plt 
from pylab import *
s= np.linspace(-4,4,1000)

def FourierTransform(a):
	Ff = (1/(a**2))*np.exp((-np.pi*(s**2))/(a**2))
	return Ff
Ff1 = FourierTransform(1)
Ff2 = FourierTransform(1.2)
Ff3 = FourierTransform(1.3)
Ff4 = FourierTransform(2)
Ff5 = FourierTransform(2.3)

plt.plot(s,Ff1,'b',label='In this plot,a = 1')
plt.plot(s,Ff2,'r',label='In this plot,a = 1.2')
plt.plot(s,Ff3,'k',label='In this plot,a = 1.3')
plt.plot(s,Ff4,'m',label='In this plot,a = 2')
plt.plot(s,Ff5,'y',label='In this plot,a = 2.3')
plt.legend()
plt.ylabel('Ff(s)')
plt.xlabel('s')
plt.grid()
plt.show()
