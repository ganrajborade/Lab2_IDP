from pylab import *
import matplotlib.pyplot as plt 
import numpy as np

def function1(a,Rtime,Rfreq):
	L = 1/Rtime
	t = np.arange(-L/2,L/2,Rtime) # time vector
	y = (np.exp((-a**2)*np.pi*((t)**2)))*(1-((2*np.abs(t))/L))
	return t,y,L

def function2(a,Rtime,Rfreq):
	L = 1/Rtime
	t = np.arange(-L/2,L/2,Rtime) # time vector
	y = (np.exp((-a**2)*np.pi*((t)**2)))*(np.sin(2*np.pi*t/L))**2
	return t,y,L

def DFT_cal_for_givenFunction1(a,Rtime,Rfreq):
	Fs = 1.0/Rtime; # sampling rate
	t,y,L = function1(a,Rtime,Rfreq)
	N = len(y)# length of the signal
	Rfreq = 1/(N*Rtime)
	k = np.arange(N)
	frq = k*Rfreq # two sides frequency range
	frq = frq[range(int(N/2))] # one side frequency range
	Y = np.abs(np.fft.fft(y)/N) # fft computing and normalization
	Y = Y[range(int(N/2))]
	return L,t,y,N,frq,Y

def DFT_cal_for_givenFunction2(a,Rtime,Rfreq):
	Fs = 1.0/Rtime; # sampling rate
	t,y,L = function2(a,Rtime,Rfreq)
	N = len(y)# length of the signal
	Rfreq = 1/(N*Rtime)
	k = np.arange(N)
	frq = k*Rfreq # two sides frequency range
	frq = frq[range(int(N/2))] # one side frequency range
	Y = 2*np.abs(np.fft.fft(y)/N) # fft computing and normalization
	Y = Y[range(int(N/2))]
	return L,t,y,N,frq,Y

L1,t1,y1,N1,frq1,Y1 = DFT_cal_for_givenFunction1(1,1/100,1/3)
L2,t2,y2,N2,frq2,Y2 = DFT_cal_for_givenFunction2(1,1/100,1/3)

subplot(2, 2,1)
plt.plot(t1,y1,'b',label='function1 in reference to w2(t) when a = 1')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(-2,2)
plt.grid()
plt.legend(loc = 'best')

subplot(2,2,2)
plt.plot(frq1,abs(Y1),'r',label = 'DFT(approximation of Ff1(s)) when a = 1 and Rtime = 1/100 s,Rfreq = 1/3Hz,f>0Hz') # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y1(freq)|')
plt.xlim(0,8)
plt.grid()
plt.legend()
#plt.title('fig. for (f) when a = 1 and Rtime = 1/100 s,Rfreq = 1/3Hz ')
print('In Row1,there are two plots for function1 and its Ff1(s) approx,we observe that we have taken a = 1 and Rtime = 1/100 s ,Rfreq = 1/3 Hz,N = 300')
print('----------------------------------------------')

subplot(2, 2,3)
plt.plot(t2,y2,'b',label='function2 in reference to w3(t) when a = 1')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(-2,2)
plt.grid()
plt.legend(loc = 'best')

subplot(2,2,4)
plt.plot(frq2,abs(Y2),'r',label = 'DFT(approximation of Ff2(s)) when a = 1 and Rtime = 1/100 s,Rfreq = 1/3Hz,f>0Hz') # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y2(freq)|')
plt.xlim(0,8)
plt.grid()
plt.legend()
#plt.title('fig. for (f) when a = 1 and Rtime = 1/100 s,Rfreq = 1/3Hz ')
print('In Row2,there are two plots for function2 and its F2(s) approx,we observe that we have taken a = 1 and Rtime = 1/100 s ,Rfreq = 1/3 Hz,N = 300')
print('----------------------------------------------')

print("Here we are normalzing fft calculation to 2*(1/N) times the fft value,because the value of fft becomes very large if we don't do these,just for understanding purpose we are taking this.")
print('----------------------------------------------')
print('Also we can say that we are considering our calculations for t ranging from -L/2 to L/2 , and for t>L/2 or t<-L/2,the value of the function f(t)w2(t) and f(t)w3(t) = 0')
plt.show()
