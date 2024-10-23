# deepfake-detection-model-for-images-and-videos
A deep learning-based approach using CNNs to identify deepfake content. Trained on diverse datasets, the model detects subtle inconsistencies in images and video frames, helping combat misinformation and verify digital media authenticity.
README for Deepfake Detection Project
Introduction
This repository contains the code and resources for detecting deepfake images and videos. The
project leverages machine learning techniques, specifically Convolutional Neural Networks
(CNNs), to accurately identify manipulated media. The model has achieved an accuracy of 94% in
distinguishing between real and deepfake content.
Methodology
1. Data Collection: Gathered a comprehensive dataset of real and deepfake images/videos.
   
   for dataset:please refer to the shared Google Drive link provided below.
   
               https://drive.google.com/drive/folders/1j2Ec01IxuepGL5PjQjTkuGb31MUgTNL6?usp=sharing
4. Feature Extraction: Employed advanced techniques to extract relevant features.
5. Model Selection: Chose a Convolutional Neural Network (CNN) for its eﬀectiveness in image
analysis.
6. Training: Trained the model on labeled data to recognize patterns.
7. Testing: Evaluated the model's performance on unseen data.
Results
1. -
Accuracy: The model achieved an accuracy of in distinguishing between real and deepfake
content.
Steps to Run the Code
copy the codes manually or import the codes
2. Prepare the Dataset/Upload the data
Place your dataset in the `data` directory. Ensure the structure is as follows:
```
train/
├── real/
│ ├── real_image1.jpg
│ └──
...
└── fake/
├── fake_image1.jpg
└──
…
test/
├── real/
│ ├── real_image1.jpg
│ └──
...
└── fake/
├── fake_image1.jpg
└──
…
validation/
├── real/
│ ├── real_image1.jpg
│ └──
...
└── fake/
├── fake_image1.jpg
└── ... ```
3. Update the path of data sets
4. Train the Model
Run the training script to train the model
5. Save the Trained Model
The trained model will be saved in the `models` directory. Ensure to specify the path correctly in
your prediction scripts.
with open('model_f_real_pickle_final', 'wb') as f:
pickle.dump(model, f)
Upload and Check Image for Deepfake
1. Load the Trained Model
with open('model_f_real_pickle_final', 'rb') as f: # Update this path
model_saved = pickle.load(f)
2. Upload an Image
Use the provided script to upload an image and check whether it is fake or not:
# Add an image of your choice
image_path = ‘image_path.png/jpg’ # Update this path
image = load_and_preprocess_image(image_path)
# Plot and predict for the new image
plt.figure(figsize=(6, 6))
plt.imshow(image.numpy())
predicted_class, confidence = pred(model_saved, image.numpy())
plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
plt.axis("oﬀ")
plt.show()
Get The Results….
Authors
- K Srinidhi
- Diwakar Kamat
- Vedant Vanaparthi
Contact Information
For more information, please contact.
• Diwakar kamat
diwakarkamath777@gmial.com
• K Srinidhi
ksrinidhi1404@gmail.com
---
This README provides a comprehensive guide to setting up and using the deepfake detection
project. It covers everything from uploading the code and dataset to running predictions on
uploaded images.
