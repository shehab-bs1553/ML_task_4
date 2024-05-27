## Project's Title:

     Audio processing and image processing using OpenCV and Python.

## Project Description: 

    The project involves two main sections:
  
      1.Computer Vision Section:
        * Basic image processing operations using OpenCV.
        * Operations include grayscale conversion, resizing, edge detection, histogram equalization, and thresholding.
        * Selected images are saved in the Selected_Image_folder for performing the above task on the images.
        * Processed images are saved in the cv_output directory.
        
      2.Audio Processing Section:
        * Fundamental audio processing techniques using a chosen audio processing library.
        * Operations include waveform plotting, spectrogram computation, and feature extraction
         (MFCC, chroma features, spectral contrast, zero-crossing rate, and spectral roll-off).
        * Selected audios are saved in the Selected_audio_file for performing the above task on the audio file.
        * Plots are saved as images in the audio_plots directory
        * Features are saved in the audio_features.csv CSV file.


## To download the project : 

For clone the repository into your file open the cmd and write :
    
    https://github.com/shehab-bs1553/ML_task_4.git

## To active the virtual environment: 
  
   #### 1.Execute this first run the command in your commend prompt 
                    
                    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass 

  #### 2. After that run the below command-   
  
                     .\venv\Scripts\Activate 
## To install all the dependencies:

        pip install -r requirements.txt


## To run the python script : 
        
1. For image processing task write -
               
                python Code/image.py
2. For audio processing task write - 
        
                python Code/audio.py
