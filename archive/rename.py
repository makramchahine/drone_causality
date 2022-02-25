import os
import shutil

input_dir = 'C:/Users/MIT Driverless/Documents/AirSim/2021-04-24-17-00-24/images'
output_dir = 'C:/Users/MIT Driverless/Documents/deepdrone/video_run'

for i, f in enumerate(os.listdir(input_dir)):
    old_path = f'{input_dir}/{f}'
    new_path = f'{output_dir}/img_{i:05}.png'
    print(old_path)
    print(new_path)
    shutil.copyfile(old_path, new_path)

