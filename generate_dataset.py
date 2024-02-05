#!/usr/bin/env python3
import random 
import subprocess


# 20 000 images

for i in range(1, 20):
	to_call = [
		"python3",'single_video_pybullet.py',
		'--spp','1000',
		'--nb_frames', '10',
		'--nb_objects', str(int(random.uniform(1,1))),
		'--scale', '0.001',
		'--objs_folder_distrators', 'fruit_distractors_2/',
		'--outf',f"/home/workstation/dope_utils/fruit_dataset/{str(i).zfill(3)}",
		'--path_single_obj', 'models/santal_ace/santal_centered.obj',
		'--nb_distractors', str(int(random.uniform(5,10))),
		'--width', '640',
		'--height','480',
		'--skip_frame', '10',
		
	]
	subprocess.call(to_call)


	

