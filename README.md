# Pelion SageMaker Notebook Example

This repo contains an example Jupyter notebook, a source py file and some test images to be imported and used within AWS Sagemaker to demonstrate a typical Data Scientist's model "train"/"compile"/"run"/"analyze"/"(re)train".... cycle with some basic image data. 

The notebook  will utilize Pelion Edge as the Sagemaker Edge Agent endpoint for running the compiled models on NVidia Jetson NX Xavier. 

### Installation

1). copy all contents of this repo into your Sagemaker notebook

2). Select a Tensorflow 1.15/Python 3.6 CPU kernel and launch it

3). Before continuing, you must also have setup:

	a). Pelion Edge Running on NVidia Xavier (https://github.com/PelionIoT/XavierPelionEdge)
	b). Install https://github.com/DougAnsonAtARM/pelion-sagemaker-controller on your Xavier
			b1). You will need to configure the AWS credentials for 
			     pelion-sagemaker-controller per instructions above
	c). Create an Application Key in your Pelion account 
	d). Note the Pelion Device ID for your Xavier Edge Gateway in Pelion

Both c) and d) will be used to configure some of the notebook initially... 
