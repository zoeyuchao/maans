import os
import imageio

def get_folders(dir, folders):
    get_dir = os.listdir(dir)
    for i in get_dir:          
        sub_dir = os.path.join(dir, i)
        if os.path.isdir(sub_dir): 
            folders.append(sub_dir) 
            get_folders(sub_dir, folders)

gif_dir = "/home/yuchao19/project/onpolicy/onpolicy/scripts/results/Habitat/mappo/debug/run17/gifs"

print("generating gifs....")
folders = []
get_folders(gif_dir, folders)
filer_folders = [folder for folder in folders if "all" in folder or "merge" in folder]

for folder in filer_folders:
    image_names = sorted(os.listdir(folder))
    frames = []
    for image_name in image_names:
        if image_name.split('.')[-1] == "gif":
            continue
        image_path = os.path.join(folder, image_name)
        frame = imageio.imread(image_path)
        frames.append(frame)

    imageio.mimsave(str(folder) + '/render.gif', frames, duration=0.1)
print("done!")