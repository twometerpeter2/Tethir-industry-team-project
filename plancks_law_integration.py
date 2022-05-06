from re import T
import numpy as np
from scipy.integrate import quad
import pandas as pd
import matplotlib.pyplot as plt

# class plannks_law_int():

#     def __init__(self, temperature, wavelength, wavelength_upper, wavelength_lower):

#         self._plancks_c = 6.63e-34
#         self._speed_light = 3e8
#         self._boltzmann_c = 1.38e-23

#         self._temperature = temperature
#         self._x = wavelength
#         self._wavelength_upper = wavelength_upper
#         self._wavelength_lower = wavelength_lower

#     def plancks_law(self):

#         B = ((2 * self._plancks_c * (self._speed_light ^ 2)) / (self._x) ^ 5) * (1 / (np.exp((self._plancks_c * self._speed_light) / (self._x * self._boltzmann_c * self._temperature)) - 1))

#         return B

#     def integration(self):

#         result, err = quad(plannks_law_int.plancks_law(self), self._wavelength_lower, self._wavelength_upper)

h = 6.63e-34
c = 3e8
k = 1.38e-23

wav_upper = 4.48e-6 + 310e-9
wav_lower = 4.48e-6 - 310e-9

def plancks_law(wav, Temp):

    intensity = 2.0 * h * c ** 2 / ((wav ** 5) * (np.exp(h * c/ (wav * k * Temp)) - 1.0) )
    return intensity

Temp = 350

res, err = quad(plancks_law, wav_lower, wav_upper, args = Temp)

print(res)

wavelengths = np.arange(4e-9, 2e-5, 1e-9)

intensity100 = plancks_law(wavelengths, 100)
intensity150 = plancks_law(wavelengths, 150)
intensity200 = plancks_law(wavelengths, 200)
intensity250 = plancks_law(wavelengths, 250)
intensity300 = plancks_law(wavelengths, 300)
intensity350 = plancks_law(wavelengths, 350)


# print(err)

plt.plot(wavelengths * 1e9, intensity100)
plt.plot(wavelengths * 1e9, intensity150)
plt.plot(wavelengths * 1e9, intensity200)
plt.plot(wavelengths * 1e9, intensity350)
plt.plot(wavelengths * 1e9, intensity300)
plt.plot(wavelengths * 1e9, intensity350)
plt.xlabel('Inntensity', fontsize=18)
plt.ylabel('Wavelength, 1e7', fontsize=16)

# plt.axvline(x = 5e3, color = "b")

# plt.show()
# N = 10000
# x = np.arange(N)

# y = 0.5 - 0.5 * np.cos((2*np.pi*x)/(N-1))
# y1 = (np.sin((np.pi*x)/(N-1)))**2

# plt.plot(x,y)
# plt.plot(x, y1)