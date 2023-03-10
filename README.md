# Outdoor Image Classification

![image](https://user-images.githubusercontent.com/100685852/210071325-4b1cdaed-b75c-48cf-88ad-30f4d6238369.png)


## About Dataset
### Context
These are images of natural Scenes around the world.

### Content
This Data contains around 25k images of size 150x150 distributed under 6 categories.
{'buildings' : 0,
'forest': 1,
'glacier': 2,
'mountain': 3,
'sea':4,
'street':5 }

The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction.
This data was initially published on https://datahack.analyticsvidhya.com by Intel to host a Image classification Challenge.

### Acknowledgements
Thanks to https://datahack.analyticsvidhya.com for the challenge and Intel for the data.

Photo by Temi Iwalaiye on Pulse Nigeria.

### Goal:
To build a Neural Network that can classify these images accurately.

## Directions
1. Clone this repo.

2. Open your docker desktop. Build docker:

       docker build -t scenes_model .
       
3. Log into AWS

   ```aws configure```
   
4. Create repository

   ```aws ecr create-repository --repository-name scene_model_tflite.images```
   
5. Tag the image

   ```docker tag scenes_model:latest <URI>```
   
6. Push the image to AWS
   
   ```docker push <URI>```

