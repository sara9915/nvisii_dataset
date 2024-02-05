#!/usr/bin/env python3

import os
import cv2
import glob
import math
import time
import nvisii as visii
import numpy as np
import pybullet as p
import random
import argparse
import colorsys
import subprocess
from math import acos
from math import sqrt
from math import pi
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument(
    '--spp',
    default=800,
    type=int,
    help = "number of sample per pixel, higher the more costly"
)
parser.add_argument(
    '--width',
    default=500,
    type=int,
    help = 'image output width'
)
parser.add_argument(
    '--height',
    default=500,
    type=int,
    help = 'image output height'
)
# TODO: change for an array
parser.add_argument(
    '--objs_folder_distrators',
    default='google_scanned_models/',
    help = "object to load folder"
)
parser.add_argument(
    '--objs_folder',
    default='models/',
    help = "object to load folder"
)
parser.add_argument(
    '--path_single_obj',
    default=None,
    help='If you have a single obj file, path to the obj directly.'
)
parser.add_argument(
    '--scale',
    default=1,
    type=float,
    help='Specify the scale of the target object(s). If the obj mesh is in '
         'meters -> scale=1; if it is in cm -> scale=0.01.'
)

parser.add_argument(
    '--skyboxes_folder',
    default='dome_hdri_haven/',
    help = "dome light hdr"
)
parser.add_argument(
    '--nb_objects',
    default=1,
    type = int,
    help = "how many objects"
)
parser.add_argument(
    '--nb_distractors',
    default=1,
    help = "how many objects"
)
parser.add_argument(
    '--nb_frames',
    default=2000,
    help = "how many frames to save"
)
parser.add_argument(
    '--skip_frame',
    default=100,
    type=int,
    help = "how many frames to skip"
)
parser.add_argument(
    '--noise',
    action='store_true',
    default=False,
    help = "if added the output of the ray tracing is not sent to optix's denoiser"
)
parser.add_argument(
    '--outf',
    default='output_example/',
    help = "output filename inside output/"
)
parser.add_argument('--seed',
    default = None,
    help = 'seed for random selection'
)

parser.add_argument(
    '--interactive',
    action='store_true',
    default=False,
    help = "make the renderer in its window"
)

parser.add_argument(
    '--motionblur',
    action='store_true',
    default=False,
    help = "use motion blur to generate images"
)

parser.add_argument(
    '--box_size',
    default=0.5,
    type=float,
    help = "make the object movement easier"
)

parser.add_argument(
    '--focal-length',
    default=None,
    type=float,
    help = "focal length of the camera"
)

parser.add_argument(
    '--visibility-fraction',
    action='store_true',
    default=False,
    help = "Compute the fraction of visible pixels and store it in the "
           "`visibility` field of the json output. Without this argument, "
           "`visibility` is always set to 1. Slows down rendering by about "
           "50 %%, depending on the number of visible objects."
)

parser.add_argument(
    '--debug',
    action='store_true',
    default=False,
    help="Render the cuboid corners as small spheres. Only for debugging purposes, do not use for training!"
)

opt = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(f'{opt.outf}'):
    print(f'folder {opt.outf}/ exists')
else:
    os.makedirs(f'{opt.outf}')
    print(f'created folder {opt.outf}/')

opt.outf = opt.outf #f'output/{opt.outf}'

if not opt.seed is None:
    random.seed(int(opt.seed))

# # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # #


visii.initialize(headless = not opt.interactive)

if not opt.motionblur:
    visii.sample_time_interval((1,1))

visii.sample_pixel_area(
    x_sample_interval = (.5,.5),
    y_sample_interval = (.5, .5))

# visii.set_max_bounce_depth(1)

if not opt.noise:
    visii.enable_denoiser()

if True: #opt.focal_length:
    camera = visii.entity.create(
        name = "camera",
        transform = visii.transform.create("camera"),
        camera = visii.camera.create_from_intrinsics(
            name = "camera",
            fx= 610.59326171875, #opt.focal_length,
            fy= 610.605712890625, #opt.focal_length,
            cx= 317.7075500488281, #(opt.width / 2),
            cy= 238.1421356201172, #(opt.height / 2),
            width=opt.width,
            height=opt.height
        )
    )
else:
    camera = visii.entity.create(
        name = "camera",
        transform = visii.transform.create("camera"),
        camera = visii.camera.create_perspective_from_fov(
            name = "camera",
            field_of_view = 0.785398,
            aspect = float(opt.width)/float(opt.height)
        )
    )

# data structure
random_camera_movement = {
    'at':visii.vec3(0,0,0),
    'up':visii.vec3(0,0,1),
    'eye':visii.vec3(0,0.5,1)
}
camera.get_transform().look_at(
    at = random_camera_movement['at'], # look at (world coordinate)
    up = random_camera_movement['up'], # up vector
    eye = random_camera_movement['eye'],
)

visii.set_camera_entity(camera)

# # # # # # # # # # # # # # # # # # # # # # # # #

# lets turn off the ambiant lights
# load a random skybox
skyboxes = glob.glob(f'{opt.skyboxes_folder}/*.hdr')
skybox_random_selection = skyboxes[random.randint(0,len(skyboxes)-1)]

dome_tex = visii.texture.create_from_file('dome_tex',skybox_random_selection)
visii.set_dome_light_texture(dome_tex)
visii.set_dome_light_intensity(random.uniform(1.1,2))
visii.set_dome_light_rotation(visii.angleAxis(random.uniform(-visii.pi(),visii.pi()),visii.vec3(0,0,1)))


#fingers
# mesh_finger_left = visii.mesh.create_from_file("obj2", "./Fingers/tactile_2.obj")
# finger_left = visii.entity.create(
#     name="finger_left",
#     mesh = mesh_finger_left,
#     transform = visii.transform.create("finger_left"),
#     material = visii.material.create("finger_left")
# )
# mesh_finger_right = visii.mesh.create_from_file("obj1", "./Fingers/tactile_1.obj")
# finger_right = visii.entity.create(
#     name="finger_right",
#     mesh = mesh_finger_right,
#     transform = visii.transform.create("finger_right"),
#     material = visii.material.create("finger_right")
# )
# finger_right.get_transform().set_parent(camera.get_transform())
# finger_left.get_transform().set_parent(camera.get_transform())


# quate = visii.angleAxis(visii.pi()/2,visii.vec3(1,0,0))*visii.angleAxis(+visii.pi()/24,visii.vec3(1,0,0))
# finger_left.get_transform().set_rotation(quate)
# finger_right.get_transform().set_rotation(quate)
# finger_right.get_transform().set_position(visii.vec3(0.2,-0.4,0.4))
# finger_left.get_transform().set_position(visii.vec3(-0.2,-0.4,0.4))

# finger_right.get_transform().set_scale(visii.vec3(0.1,0.1,0.1))
# finger_left.get_transform().set_scale(visii.vec3(0.1,0.1,0.1))
# finger_left.get_material().set_base_color(visii.vec3(0,0,0)) 
# finger_right.get_material().set_base_color(visii.vec3(0,0,0))   

# Floor
floor = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("mesh_floor"),
    transform = visii.transform.create("transform_floor"),
    material = visii.material.create("material_floor")
)
floor_x = random.uniform(0.4,0.7)
floor_y = random.uniform(0.4,0.7)
floor.get_material().set_base_color(visii.vec3(random.uniform(0,1),random.uniform(0,1),random.uniform(0,1))) 
floor.get_material().set_metallic(random.uniform(0,1)) 
floor.get_material().set_roughness(random.uniform(0,1))
floor.get_transform().set_scale((floor_x,floor_y,1))

# fruit box
# obj_to_load = "photo_realistics_models/plastic_box/Plastic_Fruits_Box_Small.stl"
# # texture_to_load = "photo_realistics_models/plastic_box/textures/Pine_Wood_Texture_baseColor.png"
# fruit_box = "fruit_box"
# scale_fruit_box = random.uniform(0.1,0.15)
# fruit_container = visii.entity.create(
#     name=fruit_box,
#     mesh = visii.mesh.create_from_file(fruit_box,obj_to_load),
#     transform = visii.transform.create(fruit_box),
#     material = visii.material.create(fruit_box)
# )
# fruit_container.get_transform().set_position((0,0,0))
# fruit_container.get_transform().set_scale(visii.vec3(scale_fruit_box))
# fruit_container.get_material().set_base_color(visii.vec3(random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)))  
# fruit_container.get_material().set_roughness(random.uniform(0,1))  
# fruit_container.get_material().set_alpha(1.0)   

# box_texture = visii.texture.create_from_file(fruit_box, texture_to_load)
# fruit_container.get_material().set_base_color_texture(box_texture)



# # # # # # # # # # # # # # # # # # # # # # # # #
# Lets set some objects in the scene

mesh_loaded = {}
objects_to_move = []

sample_space= [
        [-20,20],
        [-20,20],
        [-30,-2]
    ]

def adding_mesh_object(
        name, 
        obj_to_load, 
        texture_to_load, 
        model_info_path=None, 
        scale=1, 
        debug=False
    ):
    global mesh_loaded, visii_pybullet, names_to_export,floor 

    if texture_to_load is None:
        toys = load_obj_scene(obj_to_load)
        if len(toys) > 1: 
            print("more than one model in the object, \
                   materials might be wrong!")
        toy_transform = visii.entity.get(toys[0]).get_transform()
        toy_material = visii.entity.get(toys[0]).get_material()
        toy_mesh = visii.entity.get(toys[0]).get_mesh()        

        obj_export = visii.entity.create(
                name = name,
                transform = visii.transform.create(
                    name = name, 
                    position = toy_transform.get_position(),
                    rotation = toy_transform.get_rotation(),
                    scale = toy_transform.get_scale(),
                ),
                material = toy_material,
                mesh = visii.mesh.create_from_file(name,obj_to_load),
            )

        toy_transform = obj_export.get_transform()
        obj_export.get_material().set_roughness(random.uniform(0.1, 0.7))

        for toy in toys:
            visii.entity.remove(toy)

        toys = [name]
    else:
        toys = [name]

        if obj_to_load in mesh_loaded:
            toy_mesh = mesh_loaded[obj_to_load]
        else:
            toy_mesh = visii.mesh.create_from_file(name, obj_to_load)
            mesh_loaded[obj_to_load] = toy_mesh

        toy = visii.entity.create(
            name=name,
            transform=visii.transform.create(name),
            mesh=toy_mesh,
            material=visii.material.create(name)
        )

        toy_rgb_tex = visii.texture.create_from_file(name, texture_to_load)
        toy.get_material().set_base_color_texture(toy_rgb_tex)
        toy.get_material().set_roughness(random.uniform(0.1, 0.7))

        toy_transform = toy.get_transform()

    ###########################

    toy_transform.set_scale(visii.vec3(scale))
    floor_pos =  floor.get_transform().get_position()


    toy_transform.set_position(visii.vec3(floor_pos[0]+random.uniform(-floor_x/2,floor_x/2),floor_pos[1]+random.uniform(-floor_y/2,floor_y/2),floor_pos[2]) + 0.05)
    rotation_model =  visii.angleAxis(-visii.pi()/2,visii.vec3(0,1,0)) * visii.angleAxis(random.uniform(-visii.pi(),visii.pi()),visii.vec3(1,0,0))
    toy_transform.set_rotation(rotation_model)


    # add symmetry_corrected transform
    child_transform = visii.transform.create(f"{toy_transform.get_name()}_symmetry_corrected")
    # print(toy_transform.get_name())
    # print(child_transform)
    child_transform.set_parent(toy_transform)

    # store symmetry transforms for later use.
    symmetry_transforms = get_symmetry_transformations(model_info_path)

    # create physics for object
    id_pybullet = create_physics(name, mass=(0.1))

    if model_info_path is not None:
        try:
            with open(model_info_path) as json_file:
                model_info = json.load(json_file)
        except FileNotFoundError:
            model_info = {}
    else:
        model_info = {}

    visii_pybullet.append(
        {
            'visii_id': name,
            'bullet_id': id_pybullet,
            'base_rot': None,
            'model_info': model_info,
            'symmetry_transforms': symmetry_transforms
        }
    )
    gemPos, gemOrn = p.getBasePositionAndOrientation(id_pybullet)

    for entity_name in toys:
        names_to_export.append(entity_name)
        add_cuboid(entity_name, scale=scale, debug=debug)



# # # # # # # # # # # # # # # # # # # # # # # # #
# Lets set some objects in the scene

mesh_loaded = {}
objects_to_move = []

sample_space= [
        [-20,20],
        [-20,20],
        [-30,-2]
    ]

if opt.interactive:
    physicsClient = p.connect(p.GUI) # non-graphical version
else:
    physicsClient = p.connect(p.DIRECT) # non-graphical version

visii_pybullet = []
names_to_export = []
# #####
# child_transform = visii.transform.create(f"{fruit_container.get_transform().get_name()}_symmetry_corrected")
# child_transform.set_parent(fruit_container.get_transform())
# symmetry_transforms = get_symmetry_transformations(None)
# id_pybullet_fruit_box = create_physics(fruit_box, mass=1)
# model_info = {}
# visii_pybullet.append(
#     {
#         'visii_id': fruit_box,
#         'bullet_id': id_pybullet_fruit_box,
#         'base_rot': None,
#         'model_info': model_info,
#         'symmetry_transforms': symmetry_transforms
#     }
# )
# names_to_export.append(fruit_box)
# add_cuboid(fruit_box, scale=scale_fruit_box, debug=opt.debug)
# p.applyExternalForce(
#     id_pybullet_fruit_box,
#     -1,
#     [0,0,9.81],
#     [0,0,0],
#     flags=p.WORLD_FRAME
# )

# ############################


google_content_folder = glob.glob(opt.objs_folder_distrators + "*/")

for i_obj in range(int(opt.nb_distractors)):

    toy_to_load = google_content_folder[random.randint(0,len(google_content_folder)-1)]

    obj_to_load = toy_to_load + "/meshes/model.obj"
    texture_to_load = toy_to_load + "/materials/textures/texture.png"
    model_info_path_distractor = toy_to_load + '/model_info.json'
    name = "obj_"+toy_to_load.split('/')[-2] + f"_{i_obj}"
    #name = toy_to_load.split('/')[-2] + f"_{i_obj}"

    adding_mesh_object(name, obj_to_load, texture_to_load,model_info_path_distractor,debug=opt.debug)
    name_obj_defined = os.path.basename(os.path.dirname(opt.path_single_obj))

shapes_models = []
for file in os.listdir(os.path.dirname(opt.path_single_obj)):
    d = os.path.join(os.path.dirname(opt.path_single_obj),file)
    shapes_models.append(d)

if opt.path_single_obj is not None:
    for i_object in range(opt.nb_objects):
        path_model_rand = random.choice(shapes_models) + '/banana_centered_scaled.obj'
        print("Model Selected:",path_model_rand)
        model_info_path = os.path.dirname(path_model_rand) + '/model_info.json'
        row = 1
        column = 0
        if not os.path.exists(model_info_path):
            model_info_path = None
        adding_mesh_object(f"single_{name_obj_defined}_{i_object}",
                           path_model_rand,
                           None,
                           model_info_path,
                           scale=opt.scale,
                           debug=opt.debug)
    
else:
    google_content_folder = glob.glob(opt.objs_folder + "*/")

    for i_obj in range(int(opt.nb_objects)):
        toy_to_load = google_content_folder[random.randint(0, len(google_content_folder) - 1)]

        obj_to_load = toy_to_load + "/google_16k/textured.obj"
        texture_to_load = toy_to_load + "/google_16k/texture_map_flat.png"
        model_info_path = toy_to_load + "/google_16k/model_info.json"
        name = "hope_" + toy_to_load.split('/')[-2] + f"_{i_obj}"

        adding_mesh_object(name, obj_to_load, texture_to_load, model_info_path, scale=opt.scale, debug=opt.debug)


    camera_pybullet_col = p.createCollisionShape(p.GEOM_SPHERE,0.05)
    camera_pybullet = p.createMultiBody(
    baseCollisionShapeIndex = camera_pybullet_col,
    basePosition = camera.get_transform().get_position(),
)


# # # # # # # # # # # # # # # # # # # # # # # # #
def rotateCamera(value,frame_in_a_round):
    #frame_in_a_round = 50
    value = value / frame_in_a_round
    cam_pos = camera.get_transform().get_position()

    camera.get_transform().look_at(
        at = (0, 0, 0), # at position
        up = (0, 0, 1),   # up vector
        eye = (0.4 * math.cos(value * 2 * visii.pi()), 0.4 * math.sin(value * 2 * visii.pi()), cam_pos[2])   # eye position
    )
def rotateCameraElevation(value,factor):
    #factor = 30
    value = (value%factor)/factor
    offset = 0.5
    amplitude = 0.2
    value = offset + amplitude * math.sin(value * 2 * visii.pi())
    #value = 1 + ((value%30)/30)
    cam_pos = camera.get_transform().get_position()
    camera.get_transform().look_at(
        at = (0, 0, 0), # at position
        up = (0, 0, 1),   # up vector
        eye = (cam_pos[0], cam_pos[1], value)   # eye position
)
def cameraFollowBox(fruit_container):
    cam_pos = camera.get_transform().get_position()
    fruit_pos = fruit_container.get_transform().get_position()
    camera.get_transform().look_at(
        at = (0, 0, fruit_pos[2]), # at position
        up = (0, 0, 1),   # up vector
        eye = (cam_pos[0], cam_pos[1], fruit_pos[2])   # eye position
)

export_to_ndds_folder_settings_files(
    opt.outf,
    obj_names=names_to_export,
    width=opt.width,
    height=opt.height,
    camera_name='camera',
)

i_frame = -1
i_render = 0

factor = random.uniform(25,35)
frame_in_a_round = random.uniform(20,100)
#p.setGravity(0,0,-9.8)

for i_entry, entry in enumerate(visii_pybullet):
      #print("external iteration: ",i_entry)
      for j_entry, entry_j in enumerate(visii_pybullet):
        if j_entry!=i_entry:
            #print("internal iteration: ",j_entry)
            closest_points = p.getClosestPoints(entry['bullet_id'],entry_j['bullet_id'],1)
           
            if not closest_points:
                distance = 10
            else:
                distance = closest_points[0][8]
           
            if distance<0:
                #print("removing apple:")
                #print(entry_j['visii_id'])
                p.removeBody(entry_j['bullet_id'])
                names_to_export.remove(entry_j['visii_id'])
                visii.entity.remove(entry_j['visii_id'])
                del visii_pybullet[j_entry]
                j_entry = j_entry - 1

while True:
    p.stepSimulation()
    i_frame += 1
    #cameraFollowBox(fruit_container)
    rotateCamera(i_frame,frame_in_a_round)
    rotateCameraElevation(i_frame,factor)
    if i_render >= int(opt.nb_frames) and not opt.interactive:
        break

    # update the position from pybullet
    for i_entry, entry in enumerate(visii_pybullet):
        # print('update',entry)
        visii.entity.get(entry['visii_id']).get_transform().set_position(
            visii.entity.get(entry['visii_id']).get_transform().get_position(),
            previous = False
        )
        visii.entity.get(entry['visii_id']).get_transform().set_rotation(
            visii.entity.get(entry['visii_id']).get_transform().get_rotation(),
            previous = False
        )
        
        update_pose(entry)
        
    if i_frame == 0:
        continue

    if not opt.interactive:
        if not i_frame % int(opt.skip_frame) == 0:
            continue

        print(f"{str(i_render).zfill(5)}/{str(opt.nb_frames).zfill(5)}")

        visii.sample_pixel_area(
            x_sample_interval = (0,1),
            y_sample_interval = (0,1))

        visii.render_to_file(
            width=int(opt.width),
            height=int(opt.height),
            samples_per_pixel=int(opt.spp),
            file_path=f"{opt.outf}/{str(i_render).zfill(5)}.png"
        )
        visii.sample_pixel_area(
            x_sample_interval = (.5,.5),
            y_sample_interval = (.5,.5))

        visii.sample_time_interval((1,1))

        visii.render_data_to_file(
            width=opt.width,
            height=opt.height,
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="entity_id",
            file_path = f"{opt.outf}/{str(i_render).zfill(5)}.seg.exr"
        )
        segmentation_mask = visii.render_data(
            width=int(opt.width),
            height=int(opt.height),
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="entity_id",
        )
        segmentation_mask = np.array(segmentation_mask).reshape((opt.height, opt.width, 4))[:, :, 0]
        export_to_ndds_file(
            f"{opt.outf}/{str(i_render).zfill(5)}.json",
            obj_names = names_to_export,
            width = opt.width,
            height = opt.height,
            camera_name = 'camera',
            # cuboids = cuboids,
            camera_struct = random_camera_movement,
            segmentation_mask=segmentation_mask,
            compute_visibility_fraction=opt.visibility_fraction,
        )
        visii.render_data_to_file(
            width=opt.width,
            height=opt.height,
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="depth",
            file_path = f"{opt.outf}/{str(i_render).zfill(5)}.depth.exr"
        )

        i_render +=1

# subprocess.call(['ffmpeg', '-y',\
#     '-framerate', '30', "-hide_banner", "-loglevel", \
#     "panic",'-pattern_type', 'glob', '-i',\
#     f"{opt.outf}/*.png", f"{opt.outf}/video.mp4"])

visii.deinitialize()
