A1=float(input('Enter amplitude\t:'))
f1=int(input('Enter frequency\t:'))
t=np.linspace(-np.pi,np.pi,256)
y1=A1*np.sin(2*np.pi*f1*t)
A2=float(input('Enter amplitude\t:'))
f2=int(input('Enter frequency\t:'))
y2=A2*np.sin(2*np.pi*f2*t)
A3=float(input('Enter amplitude\t:'))
f3=int(input('Enter frequency\t:'))
y3=A3*np.sin(2*np.pi*f3*t)
y=y1+y2+y3
plt.plot(t,y)
plt.show()
#print(t)


sp = np.fft.fft(y)
freq = np.fft.fftfreq(y.shape[-1])
plt.plot(freq, sp.real, freq, sp.imag)
plt.show()
f=np.linspace(-3,3,256)
mod1=[np.sqrt((i.real**2+i.imag**2)) for i in sp]
mod2=np.abs(sp)
plt.plot(f,mod2)