from scipy import signal
import scipy.io as scio

def inverse_func(data4d,curve):
    led_curve = signal.resample(led_curve,8,axis=0)
    temp = np.moveaxis(data4d,-1,-2)
    shape_ = temp.shape
    temp = np.reshape(temp,(np.cumprod(shape_[:3])[2],shape_[3]))
    temp = np.linalg.solve(led_curve.transpose(), temp.transpose())
    temp = np.reshape(temp.transpose(),shape_)
    temp = np.moveaxis(temp,-1,-2)
    return temp

path = 'S3_result'
data_list = os.listdir(path)
name = '0000'
for data_name in data_list:
    if name in data_name:
        with np.load(path + '/' + data_name) as data:
            re_gt = data['re_gt']
            re_in = data['re_in']
            re_out = data['re_out']
            ref = data['ref']
        break
led_curve = scio.loadmat('data/BandsLed.mat')['BandsLed']
result = inverse_func(re_out,led_curve)
#orig = signal.resample(ref,8,axis=2)

## load most original data
fig = display_highdimdatacube(temp[:,:,:,:8],transpose=True)
fig.show()
fig_ref = display_highdimdatacube(orig[:,:,:,:8],transpose=True)
fig_ref.show()

with open('S4_result/'+data_name[:4]+f'_MAX={MAX_V}_gtpsnr={np.mean(psnr_gt):.4f}_inputpsnr={np.mean(psnr_in):.4f}_outputpsnr={np.mean(psnr_out):.4f}.npz',"wb") as f:
    np.savez(f, re_gt=re_gt,re_in=re_in, re_out=re_out, ref=ref)
