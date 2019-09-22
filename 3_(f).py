from pylab import *
import matplotlib.pyplot as plt 
import numpy as np

def Original_function(a,Rtime):
	t = np.arange(0,2,Rtime) # time vector
	y = np.exp((-a**2)*np.pi*((t)**2))
	return t,y

def DFT_cal_for_givenFunction(a,Rtime):
	Fs = 1.0/Rtime; # sampling rate
	t,y = Original_function(a,Rtime)
	N = len(y) # length of the signal
	Rfreq = 1/(N*Rtime)
	k = np.arange(N)
	frq = k*Rfreq # two sides frequency range
	frq = frq[range(int(N/2))] # one side frequency range
	Y = 2*np.abs(np.fft.fft(y)/N) # fft computing and normalization
	Y = Y[range(int(N/2))]
	return t,y,N,frq,Y

t1,y1,N1,frq1,Y1 = DFT_cal_for_givenFunction(1,1/100)
t2,y2,N2,frq2,Y2 = DFT_cal_for_givenFunction(2,1/500)
subplot(2, 2,1)
plt.plot(t1,y1,'b',label='Original signal when a = 1')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(0,2)
plt.grid()
plt.legend()

subplot(2,2,2)
plt.plot(frq1,abs(Y1),'r',label = 'DFT(approximation of Ff(s)) when a = 1 and Rtime = 1/100 s,f>0Hz') # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y1(freq)|')
plt.xlim(0,20)
plt.grid()
plt.legend()
plt.title('fig. for (f) when a = 1 and Rtime = 1/100 s ')
print('In first two plots for signal and its Ff(s) approx,we observe that we have taken a = 1 and Rtime = 1/100 s ,hence Rfreq = 1/2 Hz,N = 200')

print('*****************************************************')
subplot(2,2,3)
plt.plot(t2,y2,'b',label='Original signal when a = 2')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(0,2)
plt.grid()
plt.legend()

subplot(2,2,4)
plt.plot(frq2,abs(Y2),'r',label = 'DFT(approximation of Ff(s)) when a = 2 and Rtime = 1/500 s,f>0Hz') # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y2(freq)|')
plt.xlim(0,20)
plt.grid()
plt.legend()
plt.title('fig. for (f) when a = 2 and Rtime = 1/500 s ')
print('In last two plots for signal and its Ff(s) approx,we observe that we have taken a = 2 and Rtime = 1/500 s ,hence Rfreq = 1/2 Hz,N = 1000')
print('----------------------------------------------')
print("Here we are normalzing fft calculation to 2*(1/N) times the fft value,because the value of fft becomes very large if we don't do these,just for understanding purpose we are taking this.")
plt.show()


