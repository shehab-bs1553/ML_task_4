import random
import shutil
import cv2
import os
import numpy as np
import albumentations as A
from albumentations import RandomBrightnessContrast, HorizontalFlip, Rotate


main_directory = 'pokemon_image\Pokemon'
destination_directory = 'Selected_Image_folder'

all_items = os.listdir(main_directory)

all_folders = []
for item in all_items:
    if os.path.isdir(os.path.join(main_directory, item)):
        all_folders.append(item)
        
random.seed(10)
selected_folders = random.sample(all_folders, 10)

if os.path.exists(destination_directory) and os.path.isdir(destination_directory):
        shutil.rmtree(destination_directory)
os.makedirs(destination_directory, exist_ok=True)

def select_folder_randomly():
    for i in selected_folders:
         src_path = os.path.join(main_directory, i)
         dst_path = os.path.join(destination_directory, i)
         if os.path.exists(dst_path):
            continue
         else:
            shutil.copytree(src_path, dst_path)


select_folder_randomly()


def show(f_path):
    for i in selected_folders:
        folder_path = os.path.join(f_path, i)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg','.jpeg')):
                image = cv2.imread(file_path)
                cv2.imshow(filename, image)
                cv2.waitKey(0) 
                cv2.destroyAllWindows()


# show('Selected_Image_folder')

output_directory= 'cv_output'
main_directory = 'Selected_Image_folder'
if os.path.exists(output_directory) and os.path.isdir(output_directory):
        shutil.rmtree(output_directory)

def save_into_file(name,folder,filename,Name):
    output_folder_path = os.path.join(output_directory, folder)
    os.makedirs(output_folder_path, exist_ok=True)             
    file_name, file_extension = os.path.splitext(filename)     
    output_file_path = os.path.join(output_directory, folder, f"{file_name}"+Name+f"{file_extension}")              
    cv2.imwrite(output_file_path,name)
    

def processing_image(image,filename,folder):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
    save_into_file(gray_image,folder,filename,"_gray_image")           
    edges = cv2.Canny(gray_image, 100, 200) 
    save_into_file(edges,folder,filename,"_edges_image")

    resized_image = cv2.resize(image, (256,256))
    save_into_file(resized_image,folder,filename,"_resized_image")
  
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    save_into_file(blurred_image,folder,filename,"_blured_image")
    
    noisy_image = image + np.random.normal(0, 25, image.shape).astype(np.uint8)
    save_into_file(noisy_image,folder,filename,"_noised_image")
    
    equalized_image = cv2.equalizeHist(gray_image)
    save_into_file(equalized_image,folder,filename,"_histogram_image")
    
    ret, global_threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    save_into_file(global_threshold,folder,filename,"_global_threshold_image")              
    adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    save_into_file(adaptive_threshold,folder,filename,"_adaptive_threshold_image")
    
    rotated_90_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    save_into_file(rotated_90_clockwise,folder,filename,"_rotated_90_clockwise")
    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
    save_into_file(rotated_180,folder,filename,"_rotated_180")
    
    brightened_image = np.clip(image.astype(np.int16) + 80, 0, 255).astype(np.uint8)
    save_into_file(brightened_image,folder,filename,"_brightened_image")
    
    x, y, width, height = 0, 0, 300, 300  
    cropped_image = image[y:y+height, x:x+width]
    save_into_file(cropped_image,folder,filename,"_cropped_image")

for i in selected_folders:
        folder_path = os.path.join(main_directory, i)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(file_path)
                processing_image(image,filename,i)

#If you want to display all the output images then comment out the below function and run it again
# show('cv_output')

