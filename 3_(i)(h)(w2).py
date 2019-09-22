from pylab import *
import matplotlib.pyplot as plt 
import numpy as np

def Original_function(Rtime,Rfreq):
	L = 1/Rtime
	t = np.arange(-L/2,L/2,Rtime) # time vector
	y = (np.cos(2*np.pi*t) + 0.5*np.sin(4*np.pi*t))*(1-((2*np.abs(t))/L))
	return t,y

def DFT_cal_for_givenFunction(Rtime,Rfreq):
	Fs = 1.0/Rtime; # sampling rate
	t,y = Original_function(Rtime,Rfreq)
	N = len(y)# length of the signal
	Rfreq = 1/(N*Rtime)
	k = np.arange(N)
	frq = k*Rfreq # two sides frequency range
	frq = frq[range(int(N/2))] # one side frequency range
	Y = 2*np.abs(np.fft.fft(y)/N) # fft computing and normalization
	Y = Y[range(int(N/2))]
	return t,y,N,frq,Y

t1,y1,N1,frq1,Y1 = DFT_cal_for_givenFunction(1/100,1/3)

t2,y2,N2,frq2,Y2 = DFT_cal_for_givenFunction(1/20,1/2.4)

t3,y3,N3,frq3,Y3 = DFT_cal_for_givenFunction(1/1000,1/9.989)

t4,y4,N4,frq4,Y4 = DFT_cal_for_givenFunction(1/1200,1.5)

subplot(4, 2,1)
plt.plot(t1,y1,'b',label='g(t)w2(t) when Rtime = 1/100 s and Rfreq = 1/3 Hz')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(0,2)
plt.grid()
plt.legend()

subplot(4,2,2)
plt.plot(frq1,abs(Y1),'r',label = 'DFT(approximation of Ff(s)) when Rtime = 1/100 s,Rfreq = 1/3Hz,f>0Hz') # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y1(freq)|')
plt.xlim(0,2.5)
plt.grid()
plt.legend()
#plt.title('fig. for (f) when a = 1 and Rtime = 1/100 s,Rfreq = 1/3Hz ')
print('In Row1,there are two plots for signal and its Ff(s) approx,we observe that we have taken Rtime = 1/100 s ,Rfreq = 1/3 Hz,N = 300')
print('----------------------------------------------')



subplot(4,2,3)
plt.plot(t2,y2,'b',label='g(t)w2(t) when Rtime = 1/20 s and Rfreq = 1/2.4 Hz')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(0,2)
plt.grid()
plt.legend()

subplot(4,2,4)
plt.plot(frq2,abs(Y2),'r',label = 'DFT(approximation of Ff(s)) when Rtime = 1/20 s, Rfreq = 1/2.4 Hz,f>0Hz') # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y2(freq)|')
plt.xlim(0,2.5)
plt.grid()
plt.legend()
#plt.title('fig. for (f) when a = 2 and Rtime = 1/500 s,Rfreq = 1/4 Hz ')
print('In Row2,there are two plots for signal and its Ff(s) approx,we observe that we have taken Rtime = 1/20 s ,Rfreq = 1/2.4 Hz,N = 48')
print('----------------------------------------------')



subplot(4,2,5)
plt.plot(t3,y3,'b',label='g(t)w2(t) when Rtime = 1/1000 s and Rfreq = 1/9.989 Hz')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(0,2)
plt.grid()
plt.legend()

subplot(4,2,6)
plt.plot(frq3,abs(Y3),'r',label = 'DFT(approximation of Ff(s)) when Rtime = 1/1000 s, Rfreq = 1/9.989 Hz,f>0Hz') # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y3(freq)|')
plt.xlim(0,2.5)
plt.grid()
plt.legend()
#plt.title('fig. for (f) when a = 3 and Rtime = 1/1000 s,Rfreq = 8 Hz ')
print('In Row3,there are two plots for signal and its Ff(s) approx,we observe that we have taken Rtime = 1/1000 s ,Rfreq = 1/9.989 Hz,N = 9898')
print('----------------------------------------------')



subplot(4,2,7)
plt.plot(t4,y4,'b',label='g(t)w2(t) when Rtime = 1/1200 s and Rfreq = 1.5 Hz')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(0,0.6)

plt.grid()
plt.legend()

subplot(4,2,8)
plt.plot(frq4,abs(Y4),'r',label = 'DFT(approximation of Ff(s)) when Rtime = 1/1200 s, Rfreq = 1.5 Hz,f>0Hz') # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y4(freq)|')
plt.xlim(0,2.5)
plt.grid()
plt.legend()
#plt.title('fig. for (f) when a = 3.5 and Rtime = 1/1200 s,Rfreq = 4 Hz ')
print('In Row4,there are two plots for signal and its Ff(s) approx,we observe that we have taken Rtime = 1/1200 s ,Rfreq = 1.5 Hz,N = 1800')
print('----------------------------------------------')

print("Here we are normalzing fft calculation to 2*(1/N) times the fft value,because the value of fft becomes very large if we don't do these,just for understanding purpose we are taking this.")
print('----------------------------------------------')
print('Also we can say that we are considering our calculations for t ranging from -L/2 to L/2 , as for t>L/2 or t<-L/2,the value of the function g(t)w2(t) = 0 ,g(t)w3(t) = 0')
plt.show()
