import random
import shutil
import cv2
import os
import numpy as np
import albumentations as A
from albumentations import RandomBrightnessContrast, HorizontalFlip, Rotate


main_directory = 'pokemon_image\Pokemon'
destination_directory = 'Selected_Image_folder'
os.makedirs(destination_directory, exist_ok=True)
all_items = os.listdir(main_directory)
all_folders = [item for item in all_items if os.path.isdir(os.path.join(main_directory, item))]
random.seed(42)
selected_folders = random.sample(all_folders, 10)



# Select folder randomly from the main folder: 
def select_folder_randomly():
    for i in selected_folders:
         src_path = os.path.join(main_directory, i)
         dst_path = os.path.join(destination_directory, i)
         shutil.copytree(src_path, dst_path)
    print(f"Selected folders and their contents have been copied to {destination_directory}")
    
#This below call function should be called only once for selecting the folder randomly
# #after one call this should be comment out otherwise the code gets error 

# select_folder_randomly()

#Function for Display the images: 
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

#augmentation pipeline for adding augmetation into the images
augmentations = A.Compose([
    RandomBrightnessContrast(p=1),
    HorizontalFlip(p=1),
    Rotate(limit=90, p=1)
])


output_directory= 'cv_output'
main_directory = 'Selected_Image_folder'

def save_into_file(name,folder,filename,Name):
    output_folder_path = os.path.join(output_directory, folder)
    os.makedirs(output_folder_path, exist_ok=True)             
    file_name, file_extension = os.path.splitext(filename)     
    output_file_path_edge = os.path.join(output_directory, folder, f"{file_name}"+Name+f"{file_extension}")              
    cv2.imwrite(output_file_path_edge,name)
    
#Function for doing all the task
def processing_image(image,filename,folder):
    
    #code for gray_scale the images and find the edges of the images and also saved into the cv_output file
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
    save_into_file(gray_image,folder,filename,"_gray_image")           
    edges = cv2.Canny(gray_image, 100, 200) 
    save_into_file(edges,folder,filename,"_edges_image")
    #code for resize the images and save it
    resized_image = cv2.resize(image, (256,256))
    save_into_file(resized_image,folder,filename,"_resized_image")
    #code for blurring the images and save it
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    save_into_file(blurred_image,folder,filename,"_blured_image")
    
    
    #code for add noise into the images and save it
    noisy_image = image + np.random.normal(0, 25, image.shape).astype(np.uint8)
    save_into_file(noisy_image,folder,filename,"_noised_image")
    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)
    save_into_file(equalized_image,folder,filename,"_histogram_image")
    #add global and adaptive threshold
    ret, global_threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    save_into_file(global_threshold,folder,filename,"_global_threshold_image")              
    adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    save_into_file(adaptive_threshold,folder,filename,"_adaptive_threshold_image")
    
    augmented = augmentations(image=image)
    augmented_image = augmented['image']
    save_into_file(augmented_image,folder,filename,"_augmented_image")               
    
    
#Iterate through all the images using for loop for processing the images

for i in selected_folders:
        folder_path = os.path.join(main_directory, i)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(file_path)
                processing_image(image,filename,i)

#function call for displaying all output images
# show('cv_output')

