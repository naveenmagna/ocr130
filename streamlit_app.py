import streamlit as st
import pytesseract
import cv2
import os
import pandas as pd
import numpy as np
import tempfile
from ultralytics import YOLO
from PIL import Image

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Naveen.Pendem\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Load the YOLO model
model = YOLO(r'C:\naveen\cropping weights2.pt')

st.title("Cropped Image to CSV for Normal fonts")
#############################################################################
# # Create a Streamlit sidebar for settings
# st.sidebar.title('Settings')

# # Create a text input field for the local path
# local_path = st.sidebar.text_input('Local Path for Saving Files')

# # Check if the path is provided and exists
# if local_path:
#     # Normalize the path to ensure the correct format for the user's OS
#     local_path = os.path.normpath(local_path)

#     # Check if the path exists
#     if os.path.exists(local_path):
#         os.chdir(local_path)
#         st.sidebar.write(f"Current working directory set to: {os.getcwd()}")
#     else:
#         st.sidebar.write("Specified path does not exist.")

##################################################################################################################
# Upload multiple images
uploaded_files = st.file_uploader("Upload cropped image to get csv", type=["jpg", "png"], accept_multiple_files=True)

# Create a folder to store uploaded images for object detection
uploaded_images_folder = 'uploaded images to crop'
os.makedirs(uploaded_images_folder, exist_ok=True)

# Create an empty DataFrame to store the text data from all images
all_data = pd.DataFrame()

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Process each uploaded image (main code for non-dotted images)
        image_bytes = uploaded_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Your image processing code for non-dotted images here
        # ...

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply the Canny edge detector with the determined thresholds
        # edges = cv2.Canny(image, threshold1=50, threshold2=100)  # Adjust the factor as needed

        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789 .," -c classify_bln_numeric_mode=1 -c tessedit_char_blacklist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

        # Perform OCR on the edge-detected image
        text_data = pytesseract.image_to_string(threshold_image, config=custom_config)
        text_data = text_data.replace("  ", " ")
        text_data = text_data.replace(". ", ".")
        text_data = text_data.replace(" .", ".")
        text_data = text_data.replace(",", ".")
        text_data = text_data.replace("  .", ".")
        text_data = text_data.replace(".  ", ".")

        rows = text_data.split("\n")
        table_data = [row.split(" ") for row in rows]

        # Clean the data
        for row in table_data:
            for i, cell in enumerate(row):
                row[i] = ''.join(char for char in cell if ord(char) < 128)

        # Create a DataFrame for the current image
        df = pd.DataFrame(table_data)

        # Append the data to the all_data DataFrame
        all_data = pd.concat([all_data, df])

        # Get the original file name
        image_name = uploaded_file.name

        # Provide an option to download the CSV file with the same name as the image
        csv_filename = f"{os.path.splitext(image_name)[0]}.csv"

        # Save the DataFrame to a CSV file without the index
        df.to_csv(csv_filename, index=False, header=False)

        # Create a download button with the same name as the image
        st.download_button(
            label=f"Download CSV for {image_name}",
            data=open(csv_filename, 'rb').read(),
            key=f"{image_name}_download{i}",
            file_name=csv_filename,
        )

    # Display the processed data for all images
    st.write("Processed Data for All Images:")
    st.dataframe(all_data)

# Create a Streamlit sidebar for "dotted" images and object detection
st.sidebar.title('Dotted Images to CSV and Cropping')

# Upload multiple dotted images
dotted_uploaded_files = st.sidebar.file_uploader("Upload dotted image files to get CSV file", type=["jpg", "png"], accept_multiple_files=True)

if dotted_uploaded_files:
    for uploaded_file in dotted_uploaded_files:
        # Process each uploaded image (sidebar code for dotted images)
        image_bytes = uploaded_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Your image processing code for dotted images here (modify as needed)
        # ...

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


               
        bilateral = cv2.bilateralFilter(image, 8,1000,1000)
        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated_image = cv2.dilate(bilateral, kernel_dilation, iterations=1)
        
                # Apply erosion to remove noise
        kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eroded_image = cv2.erode(dilated_image, kernel_erosion, iterations=1)
        
        eroded_image = cv2.cvtColor(eroded_image, cv2.COLOR_BGR2GRAY)

        
        equalized_image = cv2.equalizeHist(eroded_image) 
         
          
        # median = cv2.medianBlur(laplacian, ksize=3)
        
        _, threshold_image = cv2.threshold(equalized_image, 50, 200, cv2.THRESH_BINARY )
                 
        
        language_code = 'eng'
        
        custom_config = rf'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789 .," -c classify_bln_numeric_mode=1 -l {language_code}'

        # Perform OCR on the edge-detected image
        text_data = pytesseract.image_to_string(threshold_image, config=custom_config)

        text_data = text_data.replace("  ", " ")
        text_data = text_data.replace(". ", ".")
        text_data = text_data.replace(" .", ".")
        text_data = text_data.replace(",", ".")
        text_data = text_data.replace("..", ".")
        text_data = text_data.replace("...", ".")
        text_data = text_data.replace("...", ".")
        text_data = text_data.replace("  .", ".")
        text_data = text_data.replace(".  ", ".")

        rows = text_data.split("\n")
        table_data = [row.split(" ") for row in rows]

        # Clean the data
        for row in table_data:
            for i, cell in enumerate(row):
                row[i] = ''.join(char for char in cell if ord(char) < 128)

        # Create a DataFrame for the current image
        df = pd.DataFrame(table_data)

        # Append the data to the dotted_all_data DataFrame
        all_data = pd.concat([all_data, df])

        # Get the original file name
        image_name = uploaded_file.name

        # Provide an option to download the CSV file with the same name as the image
        csv_filename = f"{os.path.splitext(image_name)[0]}.csv"

        # Save the DataFrame to a CSV file without the index
        df.to_csv(csv_filename, index=False, header=False)

        # Create a download button with the same name as the image
        st.sidebar.download_button(
            label=f"Download CSV for dotted {image_name}",
            data=open(csv_filename, 'rb').read(),
            key=f"{image_name}_download",
            file_name=csv_filename,
            )

    # Display the processed data for all dotted images (if any)
    st.sidebar.write("Processed Data for Dotted Images:")
    st.sidebar.dataframe(all_data)

# Object Detection
st.sidebar.title('Upload images for cropping')
os.chdir(r'C:\10.10.10.130')
# st.sidebar.write('Get the cropped images by accessing the path - ("C:\10.10.10.130\runs") using Run "\\10.10.10.130\c$"')
uploaded_images = st.sidebar.file_uploader("Upload images for cropping", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_images:
    image_paths = []

    for i, uploaded_image in enumerate(uploaded_images):
        # Save the uploaded image to the uploaded_images folder with its original name
        temp_image_path = os.path.join(uploaded_images_folder, uploaded_image.name)
        with open(temp_image_path, 'wb') as temp_file:
            temp_file.write(uploaded_image.read())
        image_paths.append(temp_image_path)

        # Display the name of the uploaded image
        st.sidebar.write(f"Uploaded Image {i + 1}: {uploaded_image.name}")

    if st.sidebar.button("Detect and Crop coordinate Table"):
        for i, image_path in enumerate(image_paths):
            results = model.predict(image_path, save=True, imgsz=320, conf=0.25, save_crop=True)
            st.sidebar.success(f"Object detection and cropping for Image {i + 1} completed!")
            

