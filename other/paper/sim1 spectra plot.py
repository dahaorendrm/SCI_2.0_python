import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
import scipy.io as scio

gaptvpsnr = [33.83,32.44,30.92,29.98]
gaptvssim = [0.9203,0.8957,0.8733,0.8554]
spvipsnr = [29.99,29.22,29.37,29.36]
spvissim = [0.8233,0.7834,0.7801,0.7773]
hsipsnr = [23.94,25.32,25.12,24.23]
hsissim = [0.6768,0.7173,0.7090,0.6803]

## Plot
plt.style.use('seaborn-bright')

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 4), layout='constrained')
x = range(16,41,8)
ax1.plot(x,gaptvpsnr,label='GapTV', marker='.', markersize=12)
ax1.plot(x,spvipsnr,label='PnP-SpVi', marker='.', markersize=12)
ax1.plot(x,hsipsnr,label='PnP-HSI', marker='.', markersize=12)
ax1.set_xlabel('CPF')  # Add an x-label to the axes.
ax1.set_ylabel('PSNR (dB)')  # Add a y-label to the axes.
#ax1.set_title("Selected points spectra")  # Add a title to the axes.


#fig, ax = plt.subplots(1,2,figsize=(5, 3), layout='constrained')
ax2.plot(x,gaptvssim,label='GapTV', marker='.', markersize=12)
ax2.plot(x,spvissim,label='PnP-SpVi', marker='.', markersize=12)
ax2.plot(x,hsissim,label='PnP-HSI', marker='.', markersize=12)
ax2.set_xlabel('CPF')  # Add an x-label to the axes.
ax2.set_ylabel('SSIM')  # Add a y-label to the axes.
#ax.set_title("Selected points spectra")  # Add a title to the axes.

plt.legend()
plt.show()
