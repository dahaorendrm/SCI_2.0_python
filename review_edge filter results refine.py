
import numpy as np
import tifffile as tif
import utils
import pywt
import pywt.data

data_spvi = tif.imread('/lustre/arce/X_MA/SCI_2.0_python/S1_pnp/data_paperpnp/img_n/toy2spvi_0000.tiff')
data_spvi = data_spvi-np.amin(data_spvi)
data_spvi = data_spvi/np.amax(data_spvi)*0.7
data_gaptv = tif.imread('/lustre/arce/X_MA/SCI_2.0_python/S1_pnp/data_paperpnp/img_n/toy2gaptv_0000.tiff')
#data_gaptv = data_gaptv-np.amin(data_gaptv)
#data_gaptv = data_gaptv/np.amax(data_gaptv)
data_ref = tif.imread('/lustre/arce/X_MA/SCI_2.0_python/S1_pnp/data_paperpnp/gt/toy2gaptv_0000.tiff')
new_data_ref = np.zeros_like(data_spvi)
for i in range(32):
    new_data_ref[...,i] = data_ref[...,i%8,i]


# Wavelet analysis
wl_img_ref = pywt.dwt2(new_data_ref[:,:,0],'haar')
wl_img_spvi = pywt.dwt2(data_spvi[:,:,0],'haar')
wl_img_gaptv = pywt.dwt2(data_gaptv[:,:,0],'haar')
utils.calculate_psnr(wl_img_spvi[1][2], wl_img_ref[1][2])
utils.calculate_psnr(wl_img_gaptv[1][2], wl_img_ref[1][2])

utils.calculate_psnr(wl_img_spvi[0], wl_img_ref[0])
utils.calculate_psnr(wl_img_gaptv[0], wl_img_ref[0])



edges_map_ref = edge_img(new_data_ref)
new_edges_map_ref = np.zeros_like(edges_map_ref)
new_edges_map_ref[edges_map_ref>0] = 1
edges_map_spvi = edge_img(data_spvi)
new_edges_map_spvi = np.zeros_like(edges_map_spvi)
new_edges_map_spvi[edges_map_spvi>0] = 1
edges_map_gaptv = edge_img(data_gaptv)
new_edges_map_gaptv = np.zeros_like(edges_map_gaptv)
new_edges_map_gaptv[edges_map_gaptv>0] = 1

# Compare edges
utils.calculate_psnr(new_edges_map_ref, new_edges_map_spvi)
utils.calculate_psnr(new_edges_map_ref, new_edges_map_gaptv)


# PSNR among images
utils.calculate_psnr(new_data_ref, data_spvi)
utils.calculate_psnr(new_data_ref, data_gaptv)



# Compare thresholded edges pixels
data_ref_edge = new_data_ref * new_edges_map_ref
data_spvi_edge = data_spvi * new_edges_map_ref
data_gaptv_edge = data_gaptv * new_edges_map_ref


# Compare edges pixels
data_ref_edge = new_data_ref * edges_map_ref
data_spvi_edge = data_spvi * edges_map_ref
data_gaptv_edge = data_gaptv * edges_map_ref


# PSNR among edge pixels
utils.calculate_psnr(data_ref_edge,data_spvi_edge)
utils.calculate_psnr(data_ref_edge,data_gaptv_edge)


# PSNR among edge pixels frame by frame
psnr_spvi = 0
psnr_gaptv = 0
ssim_spvi = 0
ssim_gaptv = 0
for i in range(32):
    psnr_spvi += utils.calculate_psnr(data_ref_edge[...,i], data_spvi_edge[...,i])
    psnr_gaptv += utils.calculate_psnr(data_ref_edge[...,i], data_gaptv_edge[...,i])
    ssim_spvi += utils.calculate_ssim(data_ref_edge[...,i], data_spvi_edge[...,i])
    ssim_gaptv += utils.calculate_ssim(data_ref_edge[...,i], data_gaptv_edge[...,i])
    print(utils.calculate_psnr(data_ref_edge[...,i], data_spvi_edge[...,i]))
    print(utils.calculate_psnr(data_ref_edge[...,i], data_gaptv_edge[...,i]))
    print(utils.calculate_ssim(data_ref_edge[...,i], data_spvi_edge[...,i]))
    print(utils.calculate_ssim(data_ref_edge[...,i], data_gaptv_edge[...,i]))


psnr_spvi
psnr_gaptv
ssim_spvi
ssim_gaptv

# Measure the sharpness
measure_sharpness(new_data_ref)
measure_sharpness(data_spvi)
measure_sharpness(data_gaptv)


# Measure the sharpness of edges
measure_sharpness(data_ref_edge)
measure_sharpness(data_spvi_edge)
measure_sharpness(data_gaptv_edge)



data_ref_edge = new_data_ref * new_edges_map_spvi
data_spvi_edge = data_spvi * new_edges_map_spvi
utils.calculate_psnr(data_ref_edge,data_spvi_edge)
data_gaptv_edge = data_gaptv * new_edges_map_spvi
utils.calculate_psnr(data_ref_edge,data_gaptv_edge)



def edge_img(img):
    vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
    horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]
    edges_img = np.zeros_like(img)
    n,m,d = img.shape
    for spec in range(d):
        for row in range(3, n - 2):
            for col in range(3, m - 2):
                # create little local 3x3 box
                local_pixels = img[row - 1:row + 2, col - 1:col + 2, spec]
                # apply the vertical filter
                vertical_transformed_pixels = vertical_filter * local_pixels
                # remap the vertical score
                vertical_score = np.sum(vertical_transformed_pixels)
                # apply the horizontal filter
                horizontal_transformed_pixels = horizontal_filter * local_pixels
                # remap the horizontal score
                horizontal_score = np.sum(horizontal_transformed_pixels)
                # combine the horizontal and vertical scores into a total edge score
                edge_score = (vertical_score ** 2 + horizontal_score ** 2) ** .5
                # insert this edge score into the edges image
                edges_img[row, col, spec] = edge_score * 3
        # remap the values in the 0-1 range in case they went out of bounds
        edges_img[..., spec] = edges_img[..., spec] / np.amax(edges_img[..., spec])
    return edges_img


def measure_sharpness(tdarray):
    sharpness = 0
    for idx in range(tdarray.shape[2]):
        gy, gx = np.gradient(tdarray[...,idx])
        gnorm = np.sqrt(gx ** 2 + gy ** 2)
        sharpness += np.average(gnorm)
    return sharpness/tdarray.shape[2]
