# give Python access to Blender's functionality
import json
import math

import bpy
import mathutils
from mathutils import Euler, Vector

camera_data = bpy.data.cameras.new(name='Camera')
camera_object = bpy.data.objects.new('Camera', camera_data)
bpy.context.scene.collection.objects.link(camera_object)
scene = bpy.context.scene
scene.camera = bpy.data.objects['Camera']

# store the location of current 3d cursor
saved_location = bpy.context.scene.cursor.location  # returns a vector


# give 3dcursor new coordinates
bpy.context.scene.cursor.location = Vector((0.0,0.0,-1.0))

# add a cube into the scene
bpy.ops.mesh.primitive_cube_add()
cube = bpy.context.selected_objects[0]

bpy.context.scene.cursor.location = (0.0, 0.0, -0.75)

bpy.ops.import_scene.fbx(filepath="/Users/tobbylie/Downloads/baphomet-sword.fbx")
bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
# get a reference to the currently active object
sword = bpy.context.selected_objects[1]
#print(f"Tobby{bpy.context.selecte d_objects[0]}")

#print(f"{cube=}")
# set the origin on the current object to the 3dcursor location
#bpy.ops.object.origin_set(type='ORIGIN_CURSOR')


# set 3dcursor location back to the stored location
bpy.context.scene.cursor.location = saved_location

#import bpy
#from mathutils import Matrix

## Get the object
#ob = bpy.context.active_object

with open("/Users/tobbylie/Documents/rl_tests/instances/test.json") as f:
    data = json.load(f)

# observations = data[-1]["observations"]
observations = data[-1]["observations"]
positions = [observation[0] for observation in observations]
rotations = [observation[2] for observation in observations]

# insert keyframe at frame one
#with open("/Users/tobbylie/Documents/rl_anims/positions.txt") as f:
#    lines = f.readlines()
#    positions = (float(line.strip()) for line in lines)

#with open("/Users/tobbylie/Documents/rl_anims/rotations.txt") as f:
#    lines = f.readlines()
#    rotations = (float(line.strip()) for line in lines)

print(rotations)

cube.scale.x = 0.4
cube.scale.y = 0.4
cube.scale.z = 0.4

sword.scale.x = 0.4
sword.scale.y = 0.4
sword.scale.z = 0.4

frame = 0
for (position, rotation) in zip(positions, rotations):
#    if frame == 500:
#        break
    cube.keyframe_insert("location", frame=frame)
    cube.location.x = position
    cube.location.z = -0.3
#    cube.keyframe_insert("rotation_euler", frame=frame)
#    cube.rotation_euler.y= rotation

    sword.keyframe_insert("location", frame=frame)
    sword.location.x = position
    sword.location.z = 0.0

    sword.keyframe_insert("rotation_euler", frame=frame)
    sword.rotation_euler.y = rotation
#    sword.rotation_axis_angle[1] = rotation
#    sword.rotation_euler = (0.0, rotation, 0.0)



#    for _ in range(2):
#        frame += 1
#        cube.keyframe_insert("location", frame=frame)
#        sword.keyframe_insert("location", frame=frame)
#        sword.keyframe_insert("rotation_euler", frame=frame)
    frame += 1
#    frame += 5


def update_camera(camera, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), distance=12.0):
    """
    Focus the camera to a focus point and place the camera at a specific distance from that
    focus point. The camera stays in a direct line with the focus point.

    :param camera: the camera object
    :type camera: bpy.types.object
    :param focus_point: the point to focus on (default=``mathutils.Vector((0.0, 0.0, 0.0))``)
    :type focus_point: mathutils.Vector
    :param distance: the distance to keep to the focus point (default=``10.0``)
    :type distance: float
    """
    looking_direction = camera.location - focus_point
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()
    # Use * instead of @ for Blender <2.8
    camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))

camera_object.location.z = 2.0
camera_object.location.y = 5.0
update_camera(bpy.data.objects['Camera'])

#from pathlib import Path
#Path("/Users/tobbylie/Documents/rl_tests/renders").mkdir(parents=True, exist_ok=True)
#for moves in range(1000):
#    scene.frame_set(moves)
#    bpy.context.scene.render.filepath = "/Users/tobbylie/Documents/rl_tests/renders/"+str(moves)+".png"
#    bpy.ops.render.render(use_viewport = True, write_still=True)
