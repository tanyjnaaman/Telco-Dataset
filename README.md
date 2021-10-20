# Telco Default Prediction
 Here, I explore some machine learning techniques for handling tabular data and deploy the models using the Flask API.

 # Process
 1. Literature
 
 My experience has laid more in computer vision and NLP, so I was not very familiar with machine learning and deep learnng techniques for structured tabular data. I was somewhat familiar with tree-based techniques. So, I looked through the relevant literature and later attempted to implement some of these models.

 The papers are documented in "literature". 

 2. Exploration and development of models

 Using a Jupyter Notebook to document my findings and progress, I clean and explore the dataset and develop several models and attempt to engineer features to improve performance. 
 Specifically, I have implemented an XGBoost model, a model based on ResNet and another based on the Transformer architecture.

 All three have decent model performance, with classification accuracy, precision, recall and F1-scores of about 80%. These metrics have been chosen to give a holistic picture of 
 classification performance. 

 These are documented in the "Development" folder.

 3. Deployment

 I have deployed the models using Flask. 

 # How to
 1. Using Docker

 To get this up and running, go to the docker folder. Download the repo. Then go to your terminal, and run:

    sudo docker build -t build

 This should install the relevant dependencies. The main program can be run from the main.py file in "Deployment" folder.

 2. Manual (in case Docker fails)

 Download all the files in the repo. You can find the necessary dependencies under the Docker folder, in requirements.txt. 
 The entire application is implemented in python. I recommend installing Anaconda and running it from a virtual environment. 

 After installing all the dependencies by:

    pip3 install <package>

 You can run the application by calling 
    
    main.py
