import random
import shutil
import cv2
import os
import numpy as np

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
    

select_folder_randomly()

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

#function for add Noise
def add_noise(image):
    noise = np.zeros(image.shape, np.uint8)
    cv2.randn(noise, 0, 25)
    noisy_image = cv2.add(image, noise)
    return noisy_image

output_directory= 'cv_output'
main_directory = 'Selected_Image_folder'
# def add_into_file(name):
#     output_folder_path = os.path.join(output_directory, folder)
#     os.makedirs(output_folder_path, exist_ok=True)             
#     file_name, file_extension = os.path.splitext(filename)               
#     os.path.join("output_file_path_"f"{name}") = os.path.join(output_directory, folder, f"{file_name}_edges{file_extension}")              
#     cv2.imwrite(output_file_path_edge,name)
    
#Function for doing all the task
def processing_image(image,filename,folder):
    #code for gray_scale the images and find the edges of the images and also saved into the cv_output file
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)              
    edges = cv2.Canny(gray_image, 100, 200) 
    #code for resize the images and save it
    resized_image = cv2.resize(image, (225,225))
    #code for blurring the images and save it
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    #code for add noise into the images and save it
    noisy_image = add_noise(image)
    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)
    #add global and adaptive threshold
    _, global_threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)              
    adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                   
    
    output_folder_path = os.path.join(output_directory, folder)
    os.makedirs(output_folder_path, exist_ok=True)             
    file_name, file_extension = os.path.splitext(filename)               
    output_file_path_edge = os.path.join(output_directory, folder, f"{file_name}_edges{file_extension}")              
    cv2.imwrite(output_file_path_edge,edges)
    output_file_path_gray = os.path.join(output_directory, folder, f"{file_name}_gray_image{file_extension}")              
    cv2.imwrite(output_file_path_gray,gray_image)
    output_file_path_resize = os.path.join(output_directory, folder, f"{file_name}_resized_image{file_extension}")              
    cv2.imwrite(output_file_path_resize,resized_image)
    
    output_file_path_blur = os.path.join(output_directory, folder, f"{file_name}_blurred_image{file_extension}")              
    cv2.imwrite(output_file_path_blur,blurred_image )
    output_file_path_noise = os.path.join(output_directory, folder, f"{file_name}_noise_image{file_extension}")              
    cv2.imwrite(output_file_path_noise,noisy_image)
    output_file_path_hist = os.path.join(output_directory, folder, f"{file_name}_hist_image{file_extension}")              
    cv2.imwrite(output_file_path_hist,equalized_image)
    
    global_output_file_path = os.path.join(output_directory, folder, f"{file_name}_global_threshold{file_extension}")
    cv2.imwrite(global_output_file_path, global_threshold)
    output_file_path_adap = os.path.join(output_directory, folder, f"{file_name}_adap_threshold_image{file_extension}")              
    cv2.imwrite(output_file_path_hist,adaptive_threshold)
    
#Iterate through all the images using for loop

for i in selected_folders:
        folder_path = os.path.join(main_directory, i)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(file_path)
                processing_image(image,filename,i)

#function call for displaying all output images
# show('cv_output')

