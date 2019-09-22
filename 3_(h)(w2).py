from pylab import *
import matplotlib.pyplot as plt 
import numpy as np

def Original_function(a,Rtime,Rfreq):
	L = 1/Rtime
	t = np.arange(-L/2,L/2,Rtime) # time vector
	y = (np.exp((-a**2)*np.pi*((t)**2)))*(1-((2*np.abs(t))/L))
	return t,y,L

def DFT_cal_for_givenFunction(a,Rtime,Rfreq):
	Fs = 1.0/Rtime; # sampling rate
	t,y,L = Original_function(a,Rtime,Rfreq)
	N = len(y)# length of the signal
	Rfreq = 1/(N*Rtime)
	k = np.arange(N)
	frq = k*Rfreq # two sides frequency range
	frq = frq[range(int(N/2))] # one side frequency range
	Y = 2*np.abs(np.fft.fft(y)/N) # fft computing and normalization
	Y = Y[range(int(N/2))]
	return L,t,y,N,frq,Y

L1,t1,y1,N1,frq1,Y1 = DFT_cal_for_givenFunction(1,1/100,1/3)

L2,t2,y2,N2,frq2,Y2 = DFT_cal_for_givenFunction(2,1/500,1/4)

L3,t3,y3,N3,frq3,Y3 = DFT_cal_for_givenFunction(3,1/1000,1/6.5)

L4,t4,y4,N4,frq4,Y4 = DFT_cal_for_givenFunction(3.5,1/1200,4)

subplot(4, 2,1)
plt.plot(t1,y1,'b',label='function1 when a = 1')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(-2,2)
plt.grid()
plt.legend(loc = 'best')

subplot(4,2,2)
plt.plot(frq1,abs(Y1),'r',label = 'DFT(approximation of Ff(s)) when a = 1 and Rtime = 1/100 s,Rfreq = 1/3Hz,f>0Hz') # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y1(freq)|')
plt.xlim(0,10)
plt.grid()
plt.legend()
#plt.title('fig. for (f) when a = 1 and Rtime = 1/100 s,Rfreq = 1/3Hz ')
print('In Row1,there are two plots for signal and its Ff(s) approx,we observe that we have taken a = 1 and Rtime = 1/100 s ,Rfreq = 1/3 Hz,N = 300')
print('----------------------------------------------')



subplot(4,2,3)
plt.plot(t2,y2,'b',label='function1 when a = 2')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(-2,2)
plt.grid()
plt.legend(loc = 'best')

subplot(4,2,4)
plt.plot(frq2,abs(Y2),'r',label = 'DFT(approximation of Ff(s)) when a = 2 and Rtime = 1/500 s, Rfreq = 1/4 Hz,f>0Hz') # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y2(freq)|')
plt.xlim(0,10)
plt.grid()
plt.legend()
#plt.title('fig. for (f) when a = 2 and Rtime = 1/500 s,Rfreq = 1/4 Hz ')
print('In Row2,there are two plots for signal and its Ff(s) approx,we observe that we have taken a = 2 and Rtime = 1/500 s ,Rfreq = 1/4 Hz,N = 2000')
print('----------------------------------------------')



subplot(4,2,5)
plt.plot(t3,y3,'b',label='function1 when a = 3')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(-2,2)
plt.grid()
plt.legend(loc = 'best')

subplot(4,2,6)
plt.plot(frq3,abs(Y3),'r',label = 'DFT(approximation of Ff(s)) when a = 3 and Rtime = 1/1000 s, Rfreq = 1/6.5 Hz,f>0Hz') # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y3(freq)|')
plt.xlim(0,10)
plt.grid()
plt.legend()
#plt.title('fig. for (f) when a = 3 and Rtime = 1/1000 s,Rfreq = 8 Hz ')
print('In Row3,there are two plots for signal and its Ff(s) approx,we observe that we have taken a = 3 and Rtime = 1/1000 s ,Rfreq = 1/6.5 Hz,N = 650')
print('----------------------------------------------')



subplot(4,2,7)
plt.plot(t4,y4,'b',label='function1 when a = 3.5')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.xlim(-0.25,0.25)
plt.grid()
plt.legend(loc = 'best')

subplot(4,2,8)
plt.plot(frq4,abs(Y4),'r',label = 'DFT(approximation of Ff(s)) when a = 3.5 and Rtime = 1/1200 s, Rfreq = 4 Hz,f>0Hz') # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y4(freq)|')
plt.xlim(0,10)
plt.grid()
plt.legend(loc = 'best')
#plt.title('fig. for (f) when a = 3.5 and Rtime = 1/1200 s,Rfreq = 4 Hz ')
print('In Row4,there are two plots for signal and its Ff(s) approx,we observe that we have taken a = 3.5 and Rtime = 1/1200 s ,Rfreq = 4 Hz,N = 300')
print('----------------------------------------------')

print("Here we are normalzing fft calculation to 2*(1/N) times the fft value,because the value of fft becomes very large if we don't do these,just for understanding purpose we are taking this.")
print('----------------------------------------------')
print('Also we can say that we are considering our calculations for t ranging from -L/2 to L/2 , and for t>L/2 or t<-L/2,the value of the function f(t)w2(t) = 0')
plt.show()



































