import bpy
import math
import os


def renderViews(filepath, savepath):
    #reset
    bpy.ops.object.delete(use_global=False)

    # Import model------------------------------------
    bpy.ops.import_scene.obj(filepath=filepath, filter_glob='*.obj')
    
    obj = bpy.context.selected_objects[-1]


    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')

    # put obj to the orignial point
    obj.location = (0, 0, 0)

    #resize the model
    bbox = obj.bound_box
    width = (bbox[6][0] - bbox[0][0]) * obj.scale.x
    height = (bbox[6][1] - bbox[0][1]) * obj.scale.y
    depth = (bbox[6][2] - bbox[0][2]) * obj.scale.z

    scale_factor = 2 / max(width, height, depth)

    
    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))

    # refresh the background
    bpy.context.view_layer.update()

    # six output
    viewpoints = [
        {"location": (0, 0, 4), "rotation": (0, 0, 0)},
        {"location": (0.3, 0, 4), "rotation": (0, 0, 0.1)},
        {"location": (0, 4, 0), "rotation": (-math.pi / 2, 0, 0)},
        {"location": (0, 4, 1), "rotation": (-math.pi / 2 + 0.2, 0, 0)},
        {"location": (0, -4, 0), "rotation": (math.pi / 2, 0, 0)},
        {"location": (1, -4, 0), "rotation": (math.pi / 2, 0, 0.2)},
        #{"location": (-4, 0, 0), "rotation": (0, -math.pi / 2, 0)},
    ]

    for i, viewpoint in enumerate(viewpoints):
        # Creating A New Camera Angle
        scene = bpy.context.scene
        cam = bpy.data.cameras.new("Camera")
        cam.lens = 50
        # create the second camera object
        cam_obj = bpy.data.objects.new("Camera", cam)
        # set camera
        cam_obj.location = viewpoints[i]["location"]
        cam_obj.rotation_euler = viewpoints[i]["rotation"]
        scene.collection.objects.link(cam_obj)
        # Set the Camera to active camera
        bpy.context.scene.camera = bpy.data.objects["Camera"]

        # add light
        light_data = bpy.data.lights.new(name="New Light", type='POINT')
        light_data.energy = 50  # strength of light
        light_object = bpy.data.objects.new(name="New Light", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        # locate light
        light_object.location = viewpoints[i]["location"]  

        # save image
        if i%2 == 0:
            FILE_PATH = os.path.join(savepath, 'pair0')
            objname = os.path.basename(filepath).split('.')[0]
            FILE_PATH = os.path.join(FILE_PATH, f"{objname}_{i}.png")  # f"{obj.name}_{i}.png"
        else:
            FILE_PATH = os.path.join(savepath, 'pair1')
            objname = os.path.basename(filepath).split('.')[0]
            FILE_PATH = os.path.join(FILE_PATH, f"{objname}_{i}.png")  # f"{obj.name}_{i}.png"
        
        
        
        scene.render.resolution_x = 640  # resize
        scene.render.resolution_y = 640  
        bpy.context.scene.camera = cam_obj
        bpy.context.scene.render.filepath = FILE_PATH
        bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    filepath = r'C:\Users\17197\cs\cp341_superglue\SuperGlue-pytorch\data\shrec_16'
    savepath = r'C:\Users\17197\cs\cp341_superglue\SuperGlue-pytorch\data\shrec_16_val'
    files = []
    files += [os.path.join(filepath, f) for f in os.listdir(filepath)]
    for file in files[:20]:
        file_path = os.path.join(file,'train')
        file_path = os.path.join(file_path, os.listdir(file_path)[5])
        renderViews(file_path, savepath)