import open3d as o3d
import os
import nvisii
import noise
import random
import numpy as np
import PIL
from PIL import Image
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def opt(): return None


opt.spp = 1024
opt.width = 640
opt.height = 480
opt.noise = True

# # # # # # # # # # # # # # # # # # # # # # # # #

nvisii.initialize(headless=False, verbose=True)
# nvisii.set_dome_light_intensity(1)
# nvisii.set_dome_light_color(nvisii.vec3(1, 1, 1))

if not opt.noise is True:
    nvisii.enable_denoiser()

camera = nvisii.entity.create(
    name="camera",
    transform=nvisii.transform.create("camera"),
    camera=nvisii.camera.create_from_intrinsics(
        name="camera",
        fx=610.59326171875,  # opt.focal_length,
        fy=610.605712890625,  # opt.focal_length,
        cx=317.7075500488281,  # (opt.width / 2),
        cy=238.1421356201172,  # (opt.height / 2),
        width=opt.width,
        height=opt.height
    )
)
altezza = 0.5
camera.get_transform().look_at(
    at=(0, 0, altezza),
    up=(0, 0, 1),
    eye=(0, 0, 0)
)
print(camera.get_transform().get_position())
print(camera.get_transform().get_rotation())
nvisii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

# # Create a scene to use for exporting segmentations
# sigma = 0.5
# # apple
# cad_dimension = [0.09756118059158325, 0.08538994193077087,0.09590171277523041] #x, y, z, cuboid dimensions [m]
# obj_to_load = "models/Apple/Apple_4K/food_apple_01_4k.gltf"
# obj_name = "apple"
# scale_obj = (1-sigma)/1
# for i in range(3):cad_dimension[i] = cad_dimension[i]*scale_obj
# obj_mesh = nvisii.entity.create(
#     name=obj_name,
#     mesh=nvisii.mesh.create_from_file(obj_name, obj_to_load),
#     transform=nvisii.transform.create(obj_name),
#     material=nvisii.material.create(obj_name)
# )

# obj_mesh.get_transform().set_parent(camera.get_transform())
# obj_mesh.get_transform().set_position(
#     (scale_obj*0.2, scale_obj*(0.2) - cad_dimension[1]/2, -(1 - sigma)))
# # obj_mesh.get_transform().set_position(
# #     (scale_obj*0.2, scale_obj*(0.2) - cad_dimension[1]/2, -(1 - sigma)))
# obj_mesh.get_transform().set_rotation(nvisii.quat(0.09503825016141965, -0.2553379209941138,-0.6285200770785785,0.7285140971990843))
# obj_mesh.get_transform().set_scale(nvisii.vec3(scale_obj))

# Create a scene to use for exporting segmentations
floor = nvisii.entity.create(
    name="floor",
    mesh = nvisii.mesh.create_plane("floor"),
    transform = nvisii.transform.create("floor"),
    material = nvisii.material.create("floor")
)
floor.get_transform().set_position((0,0,altezza))
floor.get_transform().set_scale((0.05,0.05,0.05))
#floor.get_material().set_roughness(1.0)



# # # # # # # # # # # # # # # # # # # # # # # # #

# nvisii offers different ways to export meta data
# these are exported as raw arrays of numbers

# for many segmentations, it might be beneficial to only
# sample pixel centers instead of the whole pixel area.
# to do so, call this function
nvisii.sample_pixel_area(
    x_sample_interval=(.5, .5),
    y_sample_interval=(.5, .5)
)

depth_array = nvisii.render_data(
    width=int(opt.width),
    height=int(opt.height),
    start_frame=0,
    frame_count=1,
    bounce=int(0),
    options="depth"
)

# xyz_data = depth_array = nvisii.render_data(
#     width=int(opt.width),
#     height=int(opt.height),
#     start_frame=0,
#     frame_count=1,
#     bounce=int(0),
#     options="position"
# )



depth_array = np.array(depth_array).reshape(opt.height, opt.width, 4)
#depth_array = np.flipud(depth_array)

# depth_array = np.array(xyz_data).reshape(opt.height, opt.width, 4)
# depth_array = np.flipud(depth_array)
# xx = depth_array[..., 1]
# yy = depth_array[..., 2]
# zz = depth_array[..., 3]
# depth_array = depth_array[..., :3]

# xx_ = np.array(xx).reshape(307200)
# yy_ = np.array(yy).reshape(307200)
# zz_ = np.array(zz).reshape(307200)


# fig_1 = plt.figure()
# ax1 = fig_1.add_subplot(projection='3d')
# ax1.scatter(xx_, yy_,zz_, marker='o')
# plt.show()


      

# save the segmentation image


def convert_from_uvd(v, u, d, fx, fy, cx, cy):
    # d *= self.pxToMetre
    print(cx)
    print(cy)
    print(fx)
    print(fy)
    x_over_z = -(cx - u) / fx
    y_over_z = -(cy - v) / fy
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
    return x, y, z


xyz = []
intrinsics = camera.get_camera().get_intrinsic_matrix(opt.width,opt.height)

ascisse = []
ordinate = []
depth_used = []
z_conversion = []
y_conversion = []
x_conversion = []
for i in range(opt.height):
    for j in range(opt.width):
       if(depth_array[i,j,0]>0 and depth_array[i,j,0]<altezza*2):
            ascisse.append(i)
            ordinate.append(j)
            x,y,z = convert_from_uvd(i,j, depth_array[i,j,0], 
            intrinsics[0][0], intrinsics[1][1], intrinsics[2][0],intrinsics[2][1])
            depth_used.append(depth_array[i,j,0]) 
            z_conversion.append(z)
            y_conversion.append(y)
            x_conversion.append(x)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(ascisse, ordinate, depth_used, marker='o')
ax.scatter(ascisse, ordinate,z_conversion, marker='o')
ax.set_xlabel('i')
ax.set_ylabel('j')
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.scatter(x_conversion, y_conversion, z_conversion, marker='o')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
plt.show()


# let's clean up the GPU
nvisii.deinitialize()