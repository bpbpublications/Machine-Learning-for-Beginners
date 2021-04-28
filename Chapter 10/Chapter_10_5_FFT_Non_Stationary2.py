A=float(input('Enter amplitude\t:'))
f1=int(input('Enter frequency 1\t:'))
f2=int(input('Enter frequency 2\t:'))
t=np.linspace(-np.pi, np.pi,256)
y=np.zeros(256)
for i in range(128):
    y[i]=A*np.sin(2*np.pi*f2*t[i])
for i in range(128,256):
    y[i]=A*np.sin(2*np.pi*f1*t[i])
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