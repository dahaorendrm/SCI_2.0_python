
import matplotlib.pyplot as plt
import tifffile as tif
import numpy as np

img = tif.imread('/lustre/arce/X_MA/SCI_2.0_python/resultpaper/S3/result/4D_Lego_24.tiff.tiff')
plotdata = img[143:143+3,202:202+3,:,:]
plotdata = np.mean(plotdata,(0,1,2))
plotdata = plotdata-np.amin(plotdata)
plotdata = plotdata/np.amax(plotdata)


img = tif.imread('/lustre/arce/X_MA/SCI_2.0_python/resultpaper/S0/spvi/gt/4D_Lego_24.tiff')
plotdata_orig = img[143:143+3,202:202+3,:,:]
plotdata_orig = np.mean(plotdata_orig,(0,1,2))
plotdata_orig = plotdata_orig-np.amin(plotdata_orig)
plotdata_orig = plotdata_orig/np.amax(plotdata_orig)


img = tif.imread('/lustre/arce/X_MA/SCI_2.0_python/resultpaper/S3/result/4D_Lego_24.tiff.tiff')
plotdata2 = img[217:217+3,55:55+3,:,:]
plotdata2 = np.mean(plotdata2,(0,1,2))
plotdata2 = plotdata2-np.amin(plotdata2)
plotdata2 = plotdata2/np.amax(plotdata2)


img = tif.imread('/lustre/arce/X_MA/SCI_2.0_python/resultpaper/S0/spvi/gt/4D_Lego_24.tiff')
plotdata_orig2 = img[217:217+3,55:55+3,:,:]
plotdata_orig2 = np.mean(plotdata_orig2,(0,1,2))
plotdata_orig2 = plotdata_orig2-np.amin(plotdata_orig2)
plotdata_orig2 = plotdata_orig2/np.amax(plotdata_orig2)


corr1 = np.corrcoef(plotdata, plotdata_orig)[0,1]
corr2 = np.corrcoef(plotdata2, plotdata_orig2)[0,1]


plt.style.use('seaborn-bright')

fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')
ax.plot(np.linspace(0, 24, 24),plotdata_orig,'k',label='Original P4')
ax.plot(np.linspace(0, 24, 24),plotdata_orig2,'r',label='Original P5')
ax.plot(np.linspace(0, 24, 24),plotdata,'k',label=f'Recon P4 corr={corr1:.4f}', marker='*', markersize=6, linestyle='--')
ax.plot(np.linspace(0, 24, 24),plotdata2,'r',label=f'Recon P5 corr={corr2:.4f}', marker='*', markersize=6, linestyle='--')

ax.set_title('Temporal plot')
plt.ylabel('Intensity(a.u.)')
plt.xlabel('Temporal frame')
leg = plt.legend()

plt.show()



