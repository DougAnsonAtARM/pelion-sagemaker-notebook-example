# pip install ipympl
# pip install pelion_sagemaker_controller

# Sagemaker Imports
import sagemaker
from sagemaker import get_execution_role
import boto3
import botocore
import tensorflow as tf
import numpy as np

# Image Imports
from PIL import Image
import matplotlib.pyplot as plt

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

# Pelion Sagemaker Controller API Import
from pelion_sagemaker_controller import pelion_sagemaker_controller

class MyNotebook:
    def __init__(self, api_key, device_id, endpoint_api):
        # Some misc settings we'll use in the analysis portions
        self.num_bytes = 5

        # Float formatting...
        self.float_format_str = "{:.8f}"
        
        # Float unpack format
        self.endian_format_str = '!f'
        
        # Float byte length 
        self.float_bytelen = 4
        
        # Initialize Sagemaker
        self.sagemaker_init()
        
        # Initialize Pelion Sagemaker Controller API
        self.pelion_sagemaker_controller_init(api_key, device_id, endpoint_api)
    
    # Sagemaker Init()
    def sagemaker_init(self):
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
        self.folder = 'DEMO-Sagemaker-Edge'
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
        print("Initializing Pelion Sagemaker Controller. Pelion API: " + endpoint_api + " Edge GW DeviceID: " + device_id)
        
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
        self.model.save(model_basename + '.h5')
        
        with tarfile.open(model_basename + '.tar.gz', mode='w:gz') as archive:
            archive.add(model_basename + '.h5')
        
        return model_basename + '.tar.gz'
        
    # Compile model and package/upload to S3
    def compile_model(self, created_model, target_device, model_basename, data_shape, framework):
        # Announce long winded task
        print("Beginning model compilation...")
        
        # Record the allocated model
        self.model = created_model;
        
        # Save off the model
        packaged_model_filename = self.save_model(model_basename)

        sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        keras_model_path = self.sess.upload_data(packaged_model_filename, self.bucket, self.folder)

        keras_compilation_job_name = 'Sagemaker-Edge-'+ str(time.time()).split('.')[0]
        print('Compilation job for %s started' % keras_compilation_job_name)

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

    # Package model:  keras-model-1.0.tar.gz
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
        print("Output Tensor result is located here in S3: " + output_tensor_url)
        output_tensor_filename = output_tensor_url.replace('s3://','')
        print("Copying Output Tensor File: " + output_tensor_filename + ' from S3 to notebook as ' + local_output_tensor_filename)
        with open(local_output_tensor_filename, 'wb') as f:
            self.s3_client.download_fileobj(self.bucket, output_tensor_filename, f)
    
    # We take the prediction result data and create a JSON-based output tensor           
    def create_output_tensor(self, input_filename, output_tensor_filename):
        # Read in the output tensor file
        file_size = os.path.getsize(output_tensor_filename)
        print("Prediction Output File Size: " + str(file_size) + " bytes")
        with open(output_tensor_filename, 'r') as fh:
            m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            tensor_bytearray = bytearray(m)

        # Read in the input bitmap filesize
        file_size = os.path.getsize(input_filename)
        print("Input Image File Size: " + str(file_size) + " bytes")
        input_image_tensor = tf.read_file(input_filename)
        print("")

        # Convert the input file to a input Tensor in TF...
        with tf.Session() as mysess:
            # Convert input image to numpy array
            input_image_np = mysess.run(input_image_tensor)
            # print('Input Image Tensor: ', input_image_tensor)
            print('Input Image Tensor Byte Array Length: ' + str(len(input_image_np)))
            print('Input Image Byte Array (first ' + str(self.num_bytes) + ' bytes): ' + str(input_image_np[0:self.num_bytes]) + "...")
            print("")

        # Read in the prediction result contents and convert it to a TF constant Tensor
        with open(output_tensor_filename, mode='r') as file:
            prediction_result_tensor_json_str = file.read()

            # Pelion's Sagemaker Controller packages the AWS Sagemaker Result Tensor as a JSON... so we can parse it here
            prediction_result_tensor_json = json.loads(prediction_result_tensor_json_str)
            prediction_result_tensor_json['byte_data'] = base64.decodebytes(prediction_result_tensor_json['b64_data'].encode('ascii'))
            
            # Record the input filename as well
            prediction_result_tensor_json['input_data'] = input_image_np

            # Display our Prediction Result Tensor Details
            print("Prediction Result Tensor Name: " + prediction_result_tensor_json['name'])
            print("Prediction Result Tensor Shape: " + str(prediction_result_tensor_json['shape']))
            print("Prediction Result Tensor Type: " + str(prediction_result_tensor_json['type']))
            print("Prediction Result Tensor Data (first " + str(self.num_bytes) + " of " 
                + str(len(prediction_result_tensor_json['byte_data'])) + " values): " 
                + str(prediction_result_tensor_json['byte_data'][0:self.num_bytes]) + "...")
            
            # return our raw tensor
            return prediction_result_tensor_json
    
    # Some models need to have the raw tensor bytestream converted to float32 values (re-dim)        
    def bytedata_to_float32data(self, tensor, float_byte_len):
        tensor_raw_byte_data = tensor['byte_data']
        tensor_byte_data_np = np.frombuffer(tensor_raw_byte_data, dtype=np.dtype('B'))  # raw byte data is uint8
        float_tensor_byte_array = []
        length = len(tensor_byte_data_np)
        i = 0
        while i < length:
            float_tensor_byte_array.append(
                float(self.float_format_str.format(
                    float('.'.join(str(elem) for elem in struct.unpack(self.endian_format_str, memoryview(tensor_byte_data_np[i:(i+float_byte_len)])))
                        )    
                    )
                )
            )
            i += float_byte_len

        # update our tensor
        tensor['float32_data'] = float_tensor_byte_array
        
        # display the formatted float array
        print("Prediction Result Tensor Data as Float32 (first " + str(self.num_bytes) + " of " 
                + str(len(tensor['float32_data'])) + " values): " 
                + str(tensor['float32_data'][0:self.num_bytes]) + "...")
        
        # return the tensor
        return tensor

    # invoke the Imagenet prediction decoder to analyze the prediction results
    def imagenet_prediction_analyzer(self, float_prediction_data):
        # Now, lets look at the predictions and see how accurate our model was
        pred_arr = np.expand_dims(float_prediction_data, axis=0)
        prediction_results = tf.keras.applications.imagenet_utils.decode_predictions(pred_arr)[0]

        # Lets Display the results
        print("")
        print('Top Image Detection Results from Imagnet:')
        for result in prediction_results:
            result_json = {}
            result_json['class'] = result[0]
            result_json['description'] = result[1]
            result_json['prob_percent'] = float(self.float_format_str.format(100.0 * result[2]))
            print(json.dumps(result_json)) 
            
    # Display our prediction results
    def display_results(self, input_image_filename, local_output_tensor_filename):
        # Lets create our Output Tensor
        tensor = self.create_output_tensor(input_image_filename, local_output_tensor_filename)  
        
        # For Imagenet analysis, we convert the tensor byte data to float32 data
        tensor = self.bytedata_to_float32data(tensor,self.float_bytelen)
        
        # Next lets use Imagenet to provide a prediction analysis
        self.imagenet_prediction_analyzer(tensor['float32_data'])
                
