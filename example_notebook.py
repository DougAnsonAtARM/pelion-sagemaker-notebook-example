# pip install ipympl
# pip install pelion_sagemaker_controller

# Sagemaker Imports
import sagemaker
from sagemaker import get_execution_role
import boto3
import botocore
import tensorflow as tf
import numpy as np

# Core Imports
import os
import mmap
import sys
import base64
import struct
import time
import tarfile
import time
import json

# TF Helpers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from IPython.display import Image, display

# Pelion Sagemaker Controller API Import
from pelion_sagemaker_controller import pelion_sagemaker_controller

class MyNotebook:
    def __init__(self, api_key, device_id, endpoint_api, aws_s3_folder):
        # Initialize Sagemaker
        self.sagemaker_init(aws_s3_folder)
        
        # Initialize Pelion Sagemaker Controller API
        self.pelion_sagemaker_controller_init(api_key, device_id, endpoint_api)
    
    # Sagemaker Init()
    def sagemaker_init(self,aws_s3_folder):
        print("")
        print("Initializing Sagemaker and S3...")
        self.s3 = boto3.resource('s3')
        self.s3_client = boto3.client('s3')
        for bucket in self.s3.buckets.all():
            print('Target Bucket: ' + bucket.name)

        self.role = get_execution_role()
        self.sess = sagemaker.Session()
        self.region = boto3.Session().region_name
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)

        # S3 bucket and folders for saving model artifacts.
        # Feel free to specify different bucket/folders here if you wish.
        self.bucket = self.sess.default_bucket()
        print('Default Bucket: ' + self.bucket)
        self.folder = aws_s3_folder
        self.compilation_output_sub_folder = self.folder + '/compilation-output'
        self.iot_folder = self.folder + '/iot'

        # S3 Location to save the model artifact after compilation
        self.s3_compilation_output_location = 's3://{}/{}'.format(self.bucket, self.compilation_output_sub_folder)

        # Display the S3 directories used
        print("")
        print("Compiled Models Location: " + self.s3_compilation_output_location)
        print("IoT Input/Output Folder: " + 's3://{}/{}'.format(bucket, self.iot_folder))
        
    # Pelion Sagemaker Controller Init()
    def pelion_sagemaker_controller_init(self, api_key, device_id, endpoint_api = 'api.us-east-1.mbedcloud.com'):
        print("")
        print("Initializing Pelion Sagemaker Controller. Pelion API: " + endpoint_api + " Pelion Sagemaker Edge Agent PT DeviceID: " + device_id)
        
        # Create an instance of the Controller API...
        self.pelion_api = pelion_sagemaker_controller.ControllerAPI(api_key,device_id,endpoint_api)
        
        # Sync the configuration to match our sagemaker config
        print("")
        print("Syncing Pelion Configuration to match Sagemakers...")
        self.pelion_api.pelion_set_config('awsS3Bucket',self.bucket)
        self.pelion_api.pelion_set_config('awsS3ModelsDirectory',self.compilation_output_sub_folder)
        self.pelion_api.pelion_set_config('awsS3DataDirectory',self.iot_folder)
        self.pelion_api.pelion_set_config('awsRegion',self.region)
        print("")
        print("Current Pelion Configuration:")
        print(self.pelion_api.pelion_get_config())
            
    # Save off the model
    def save_model(self, model_basename):
        # self.model.save(model_basename + '.h5')
        tf.keras.models.save_model(self.model,model_basename + '.h5')
        
        with tarfile.open(model_basename + '.tar.gz', mode='w:gz') as archive:
            archive.add(model_basename + '.h5')
        
        return model_basename + '.tar.gz'
        
    # Compile model and package/upload to S3
    def compile_model(self, created_model, target_device, model_basename, framework, data_shape):
        # Announce long winded task
        print("")
        print("Beginning model compilation...")
        
        # Record the allocated model
        self.model = created_model;
        
        # Save off the model
        packaged_model_filename = self.save_model(model_basename)

        sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        keras_model_path = self.sess.upload_data(packaged_model_filename, self.bucket, self.folder)

        keras_compilation_job_name = 'Sagemaker-Edge-'+ str(time.time()).split('.')[0]
        
        # Initiate the compilation job
        print("")
        print('Compilation job (%s) has started...' % keras_compilation_job_name)
        response = sagemaker_client.create_compilation_job(
                CompilationJobName=keras_compilation_job_name,
                RoleArn=self.role,
                InputConfig={
                    'S3Uri': keras_model_path,
                    'DataInputConfig': data_shape,
                    'Framework': framework.upper()
                },
                OutputConfig={
                    'S3OutputLocation': self.s3_compilation_output_location,
                    'TargetDevice': target_device 
                },
                StoppingCondition={
                    'MaxRuntimeInSeconds': 1900
                }
            )

        print(response)

        # Poll every 30 sec
        while True:
            response = sagemaker_client.describe_compilation_job(CompilationJobName=keras_compilation_job_name)
            if response['CompilationJobStatus'] == 'COMPLETED':
                break
            elif response['CompilationJobStatus'] == 'FAILED':
                print(str(response))
                raise RuntimeError('Compilation failed')
            print('Compiling ...')
            time.sleep(10)
        print('Done!')
        return keras_compilation_job_name

    # package up the model as tgz for transport via Pelion to Sagemaker Edge Agent service
    def package_model(self, keras_packaged_model_name, keras_model_version, keras_compilation_job_name):
        # Announce long winded task
        print("Beginning model packaging...")
        
        # Create the model_package name that we will use
        self.model_package = '{}-{}.tar.gz'.format(keras_packaged_model_name, keras_model_version)
        
        # Create the packaging job... 
        keras_packaging_job_name=keras_compilation_job_name+"-packaging"
        response = self.sagemaker_client.create_edge_packaging_job(
            RoleArn=self.role,
            OutputConfig={
                'S3OutputLocation': self.s3_compilation_output_location,
            },
            ModelName=keras_packaged_model_name,
            ModelVersion=keras_model_version,
            EdgePackagingJobName=keras_packaging_job_name,
            CompilationJobName=keras_compilation_job_name
        )

        print(response)

        # Poll every 30 sec
        while True:
            job_status = self.sagemaker_client.describe_edge_packaging_job(EdgePackagingJobName=keras_packaging_job_name)
            if job_status['EdgePackagingJobStatus'] == 'COMPLETED':
                break
            elif job_status['EdgePackagingJobStatus'] == 'FAILED':
                raise RuntimeError('Edge Packaging failed')
            print('Packaging ...')
            time.sleep(30)
        print('Done!')
        return self.model_package
    
    # Copy our prediction results back to our notebook from S3...
    def copy_results_to_notebook(self, output_tensor_url, local_output_tensor_filename):
        print("Output Tensor located here in S3: " + output_tensor_url)
        output_tensor_filename = output_tensor_url.replace('s3://','')
        print("Copying Output Tensor: " + output_tensor_filename + ' to local notebook as: ' + local_output_tensor_filename)
        with open(local_output_tensor_filename, 'wb') as f:
            self.s3_client.download_fileobj(self.bucket, output_tensor_filename, f)
    
    # Simple Image List Display with optional annotations...
    def display_images(self, my_list, most_likely_labels=None):
        for i, img in enumerate(my_list):
            display(img)
            if most_likely_labels != None:
                pred = most_likely_labels[i]
                pred_tuple = pred[0]
                pred_name = pred_tuple[1]
                pred_percent = round(pred_tuple[2]*100.0,3)
                print("Predicted Image Contents: \"" + pred_name + "\" Confidence: " + str(pred_percent) +"%")
            print("")

    # Read in a batch of images
    def read_image_batch(self, img_paths, img_height, img_width):
        img_list = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths if os.path.isfile(img_path)]
        array_list =  np.array([img_to_array(img) for img in img_list])
        return {"img":img_list, "array":array_list}
    
    # Save the input tensor for the sagemaker edge adgent PT to read in and pass to the AWS edge agent manager
    def save_input_tensor_to_s3(self, input_tensor, input_data_filename):
        # the Pelion edge agent PT expects the input tensor file to be organzed as a JSON
        input_data_json = {}
        input_data_json ['b64_data'] = base64.b64encode(input_tensor.numpy().data.tobytes()).decode('ascii')

        # Save the input tensor JSON data...
        print("Saving input tensor to file: " + input_data_filename)
        with open(input_data_filename, 'w') as f:
            f.write(json.dumps(input_data_json))
            f.close()
            
        # Upload the images in the list to S3
        print("")
        print('Uploading saved input tensor to ' + self.iot_folder + " in S3 bucket " + self.bucket + ' as: ' + input_data_filename + "...")
        print("")
        self.sess.upload_data(input_data_filename, self.bucket, self.iot_folder)
            
    # Method to Decode ResNet50 based predictions
    def decode_resnet50_predictions(self, preds, top=5, class_list_path=None):
        if len(preds.shape) != 2 or preds.shape[1] != 1000:
            raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
        class_index = json.load(open(class_list_path))
        results = []
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            result = [tuple(class_index[str(i)]) + (pred[i],) for i in top_indices]
            result.sort(key=lambda x: x[2], reverse=True)
            results.append(result)
        return results
    
    # Pull the output tensor from s3 back into the notebook and parse/read it into a numpy...
    def get_output_tensor(self, s3_filename, local_nb_filename, tensor_dtype=np.float32):
        # Copy the results back to our notebook
        print("Retrieving Output Tensor from S3: " + s3_filename + " Saving locally to: " + local_nb_filename + "...")
        self.copy_results_to_notebook(s3_filename,local_nb_filename)

        # Read in the output tensor file, convert it, then decode our predictions and display our results...
        print("Opening Output Tensor File: " + local_nb_filename + "...")
        file_size = os.path.getsize(local_nb_filename)
        print("Output Tensor File Size: " + str(file_size) + " bytes")
        with open(local_nb_filename, 'r') as file:
            # Load the JSON-based tensor from its file in our notebook... 
            json_tensor = json.loads(file.read())

            # Convert the (Pelion PT specific) JSON-based tensor to an Numpy Tensor with the intended shape and dtype
            uint8_buffer = base64.b64decode(json_tensor['b64_data'])
            output_tensor = np.frombuffer(uint8_buffer, dtype=tensor_dtype)
            output_tensor_reshaped = np.reshape(output_tensor,(json_tensor['shape'][0],json_tensor['shape'][1]))

            # Display the prediction result tensor details...
            print("")
            print("Output Tensor (Reshaped) - Shape: " + json.dumps(output_tensor_reshaped.shape) + " Type: " + str(output_tensor_reshaped.dtype))
            return output_tensor_reshaped