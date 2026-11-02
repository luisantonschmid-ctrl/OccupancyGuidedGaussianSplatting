import numpy as np
import matplotlib.pyplot as plt

evalfolder = '/home/luis/Desktop/Evaluations/'
ren_psnrpath = evalfolder + 'kitti_seq10w_psnr_masked.csv'
ren_ssimpath = evalfolder + 'kitti_seq10w_ssim_masked.csv'
mesh_psnrpath = evalfolder + 'kitti_seq10w_mesh_psnr_masked.csv'
mesh_ssimpath = evalfolder + 'kitti_seq10w_mesh_ssim_masked.csv'
comb_psnrpath = evalfolder + 'kitti_seq10w_comb_psnr_masked.csv'
comb_ssimpath = evalfolder + 'kitti_seq10w_comb_ssim_masked.csv'
data1 = np.loadtxt(ren_psnrpath, delimiter=",").flatten()
data2 = np.loadtxt(mesh_psnrpath, delimiter=",").flatten()
data3 = np.loadtxt(comb_psnrpath, delimiter=",").flatten()
data4 = np.loadtxt(ren_ssimpath, delimiter=",").flatten()
data5 = np.loadtxt(mesh_ssimpath, delimiter=",").flatten()
data6 = np.loadtxt(comb_ssimpath, delimiter=",").flatten()
print(ren_psnrpath,np.average(data1))
print(mesh_psnrpath,np.average(data2))
print(comb_psnrpath,np.average(data3))
print(ren_ssimpath,np.average(data4))
print(mesh_ssimpath,np.average(data5))
print(comb_ssimpath,np.average(data6))
# Plot all three arrays
plt.figure(figsize=(10, 5))

plt.plot(data1, label="Gaussians")
plt.plot(data2, label="Mesh")
plt.plot(data3, label="Combined")

plt.xlabel("Frame")
plt.ylabel("PSNR")
plt.title("PSNR Kitti10")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Plot all three arrays
plt.figure(figsize=(10, 5))

plt.plot(data4, label="Gaussians")
plt.plot(data5, label="Mesh")
plt.plot(data6, label="Combined")

plt.xlabel("Frame")
plt.ylabel("SSIM")
plt.title("SSIM Kitti10")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()