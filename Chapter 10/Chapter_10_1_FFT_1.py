A=float(input('Enter amplitude\t:'))
f=int(input('Enter frequency\t:'))
t=np.linspace(-np.pi, np.pi,256)
y=A*np.sin(2*np.pi*f*t)
plt.plot(t,y)
plt.show()
#print(t)

sp = np.fft.fft(y)
freq = np.fft.fftfreq(y.shape[-1]
)
plt.plot(freq, sp.real, freq, sp.imag)
plt.show()

f1=np.linspace(-3,3,256)
mod1=[np.sqrt((i.real**2+i.imag**2)) for i in sp]
mod2=np.abs(sp)
plt.plot(f1,mod2)