<details open>
<summary> <b>Project Overview: Image Classification using AWS SageMaker and Kaggle Fingers Dataset<b></summary>

In this project, we will be using AWS Sagemaker to finetune a pretrained model that can perform image classification. We will have to use Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices to finish this project. We are using the Kaggle Fingers dataset that consists on grayscale one channel images.

</details>


<details open>
<summary> <b>How It Works<b></summary>

The goal of the project is to build a model able to count fingers as well as distinguish between left and right hand.

21600 images of left and right hands fingers.

- All images are 128 by 128 pixels.
- Training set: 18000 images (14400 training images + 3600 validation images)
- Test set: 3600 images
- Images are centered by the center of mass
- Noise pattern on the background

For this project, we will use VGG16, a pretrained neural network to make fingers classification using PyTorch.

Once we have trained the model (using hyperparameter tuning), we will need to deploy the model to a Sagemaker Endpoint. To test your deployment, we also need to query the deployed model with a sample image and get a prediction.

#### Pipeline

We will have to perform tasks and use tools that a typical ML Engineer does as a part of their job. Broadly, our project has 3 main steps:

- Start with Data Preparations where the training data is put into a S3 bucket. 
- Next in Training, there is hyperparameter tuning, which leads to training the models and outputs a Profiler, Model, and Debugger report. 
- Finally in deployment there is deployment and testing. 

#### Project Pipeline

As an ML Engineer, we will need to track and coordinate the flow of data (which could be images, models, metrics etc) through these different steps. The goal of this project is to train an accurate model, but to set up an infrastructure that enables other developers to train such models.

We will go over your tasks for each of these steps in more detail over the next few pages.

Useful Links:
- [How to Finetune a Pytorch Model](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
- [How to finetune a TensorFlow Model](https://www.tensorflow.org/tutorials/images/transfer_learning)

</details>


<details open>
<summary> <b>Project Set Up and Installation<b></summary>

Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

</details>


<details open>
<summary> <b>Dataset<b></summary>

The provided dataset is the Kaggle Fingers Dataset for classification which can be found in [here](https://www.kaggle.com/datasets/koryakinp/fingers).
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

</details>


<details open>
<summary> <b>Access<b></summary>

Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data.
![bucket s3](doc/bucket_zip.PNG "width=80%")

Look at the results on the default bucket.
![bucket s3](doc/bucket_results.PNG "width=80%")



</details>

<details open>
<summary> <b>Files and Folders<b></summary>

- **doc**: contains the images referenced here.
  - *png_images*: image for reference on the document.
- **code**: contains the python files for hyperparameter optimization, training and deployment.
  - *hpo.py*: hyperparameter tunning file.
  - *train_model.py*: for debuging, profiling and deployment.
  - *requirements.txt*: if you want to install dependencies on the docker container.
- **ProfileReport**: The complete profiler and debugger output file.
- **fingers**: not listed here, has the training and test set downloaded and extracted from kaggle.
- **fingers_sorted**: not listed here, has the training, test and validation sets. 
- **Data Exploration.ipyng**: notebook containing the exploratory data analysis over the dataset and sorting.
- **Torch Model - Transfer Learning.ipynb**: notebook for developing the pipeline locally before using sagemaker PoC (Proof of concept)
- **train_and_deploy.ipynb**: notebook with the complete solution of the pipeline.
- **train_and_deploy.html**: same notebook as above but in html output.
- **train_and_deploy.pdf**: same notebook as above but in pdf format.
- **info_train.csv**: how was distributed the training set.
- **info_valid.csv**: how was distributed the validation set.
- **info_test.csv**: how was distributed the test set.
- **README.md**: This file, explanation of the project

</details>

<details open>
<summary> <b>Hyperparameter Tunning<b></summary>

Choosen Model:
- We choose a VGG16 model that is a pre-trained neural network with imagenet v1 weight files.

![CIFAR-100 Dataset](doc/VGG16.PNG "width=80%")

Overview of the Hyperparameters and their use:
- Learning Rate: it controls how much change the model in response ot the estimated error each time in the model weights; determines the step size at each iteration whule moving toward a minimun of a loss function.
- Batch Size: controls the number of training samples to work through before the model's internal parameter update.
- epochs: refers to one time passing of all training data over the network.

Training Jobs
![Training Job1](doc/tunning_job1.PNG "width=80%")

Log Metrics
![Training Job2](doc/tunning_job2.PNG "width=80%")

Tunned Hyperparameters

![Hyperparameters](doc/best_training_job.PNG "width=80%")

</details>

<details open>
<summary> <b>Debugging and Profiling<b></summary>

Over the notebook, the rules, profiler and configurations of debugger were set, also we set up the estimator and enable the hooks on the .py file to log the information.

</details>

<details open>
<summary> <b>Results<b></summary>

At first we saw the model will needing the GPU for training and also the time for training must not be needed as much too, we then switched to CPU and only for 4 epochs is enough, this gives us less cost through the sagemaker pipeline but sacrifices the time of training.

The debugger and profiler report is uploaded in this repository to check the outputs.

</details>

<details open>
<summary> <b>Model Deployment<b></summary>

Model Deployment and Query Input.
![Inference0](doc/inference0.PNG "width=80%")
![Inference1](doc/inference1.PNG "width=80%")
![Inference2](doc/inference2.PNG "width=80%")
![Inference3](doc/inference3.PNG "width=80%")

Screenshot of active endpoint in Sagemaker.
![Deployment](doc/endpoint.PNG "width=80%")

</details>


<details open>
<summary> <b>Standout Suggestions [OPTIONAL]<b></summary>

- 🚧 Package Your Model: Pack the model as a Docker Container so that it can be easily deployed. 
- 🚧 Multi-Model Endpoint: Finetune multiple (different) pretrained models and try to deploy them to the same endpoint in the form of a Multi-Model Endpoint.
- 🚧 Batch Transform: Create a batch transform that performs inference on the whole test set together.
- 🚧 Model Explainability: Use Amazon Sagemaker Clarity to make your models more interpretable.

</details>

<details open>
<summary> <b>Video Explanation<b></summary>

Fingers classification using AWS Sagemaker and Kaggle Dataset

[<img src= "" />]()


</details>

<details open>
<summary> <b>Contributing<b></summary>

Your contributions are always welcome! Please feel free to fork and modify the content but remember to finally do a pull request.

</details>

<details open>
<summary> :iphone: <b>Having Problems?<b></summary>

<p align = "center">

[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/riawa)
[<img src="https://img.shields.io/badge/telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"/>](https://t.me/issaiass)
[<img src="https://img.shields.io/badge/instagram-%23E4405F.svg?&style=for-the-badge&logo=instagram&logoColor=white">](https://www.instagram.com/daqsyspty/)
[<img src="https://img.shields.io/badge/twitter-%231DA1F2.svg?&style=for-the-badge&logo=twitter&logoColor=white" />](https://twitter.com/daqsyspty) 
[<img src ="https://img.shields.io/badge/facebook-%233b5998.svg?&style=for-the-badge&logo=facebook&logoColor=white%22">](https://www.facebook.com/daqsyspty)
[<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/riawe)
[<img src="https://img.shields.io/badge/tiktok-%23000000.svg?&style=for-the-badge&logo=tiktok&logoColor=white" />](https://www.linkedin.com/in/riawe)
[<img src="https://img.shields.io/badge/whatsapp-%23075e54.svg?&style=for-the-badge&logo=whatsapp&logoColor=white" />](https://wa.me/50766168542?text=Hello%20Rangel)
[<img src="https://img.shields.io/badge/hotmail-%23ffbb00.svg?&style=for-the-badge&logo=hotmail&logoColor=white" />](mailto:issaiass@hotmail.com)
[<img src="https://img.shields.io/badge/gmail-%23D14836.svg?&style=for-the-badge&logo=gmail&logoColor=white" />](mailto:riawalles@gmail.com)

</p>

</details>

<details open>
<summary> <b>License<b></summary>
<p align = "center">
<img src= "https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-sa.svg" />
</p>
</details>