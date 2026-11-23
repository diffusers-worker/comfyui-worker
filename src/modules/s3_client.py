import mimetypes
import os
import sys
import threading
import urllib.parse

import boto3
from boto3.s3.transfer import TransferConfig

from botocore.config import Config
from botocore.exceptions import ClientError

class S3Client:
	
	s3_client = None
	
	num_threads = 8
	
	INLINE_MIME_TYPES = {
		'image/jpeg',
		'image/png',
		'image/gif',
		'image/webp',
		'application/pdf',
	}
	
	def __init__ (self, config: dict):
		
		self.config = config
		
		self.s3_client = boto3.client (
			service_name = 's3',
			endpoint_url = self.config['endpoint'],
			aws_access_key_id = self.config['access_key_id'],
			aws_secret_access_key = self.config['secret_access_key'],
			region_name = 'auto',  # R2 uses 'auto' region mapping
			config = Config (signature_version = 's3v4'),  # Strictly enforce AWS Signature Version 4
		)
		
	def generate_upload_url (self, bucket: str, key: str, mime_type: str, **kwargs) -> str:
		
		"""
		Generates a secure Presigned PUT URL.
		The recipient (RunPod worker) can upload a raw payload directly to this key
		without holding any master cloud storage secret tokens.
		"""
		
		params = {
			'Bucket': bucket,
			'Key': key,
			'ContentType': 'application/octet-stream',  # Enforce strict binary framing
		}
		
		encoded_filename = urllib.parse.quote (key)
		
		if mime_type in self.INLINE_MIME_TYPES:
			
			# Opens in browser
			params['ResponseContentDisposition'] = f"inline; filename=\"file\"; filename*=UTF-8''{encoded_filename}"
			params['ResponseContentType'] = mime_type
		
		else:
			params['ResponseContentDisposition'] = f"attachment; filename=\"file\"; filename*=UTF-8''{encoded_filename}"
		
		if 'expires_in' not in kwargs:
			kwargs['expires_in'] = 3600
		
		return self.s3_client.generate_presigned_url (
			ClientMethod = 'put_object',
			Params = params,
			ExpiresIn = kwargs['expires_in'],
		)
	
	def generate_get_url (self, bucket: str, key: str, mime_type: str, **kwargs) -> str:
		
		"""
		Generates a secure Presigned GET URL.
		The recipient can upload a raw payload directly to this key
		without holding any master cloud storage secret tokens.
		"""
		
		params = {
			'Bucket': bucket,
			'Key': key,
		}
		
		encoded_filename = urllib.parse.quote (key)
		
		if mime_type in self.INLINE_MIME_TYPES:
			
			# Opens in browser
			params['ResponseContentDisposition'] = f"inline; filename=\"file\"; filename*=UTF-8''{encoded_filename}"
			params['ResponseContentType'] = mime_type
			
		else:
			params['ResponseContentDisposition'] = f"attachment; filename=\"file\"; filename*=UTF-8''{encoded_filename}"
		
		if 'expires_in' not in kwargs:
			kwargs['expires_in'] = 3600
		
		return self.s3_client.generate_presigned_url (
			ClientMethod = 'get_object',
			Params = params,
			ExpiresIn = kwargs['expires_in'],
		)
	
	def upload_file (self, file: str, bucket, **kwargs) -> str:
		
		if 'force' not in kwargs:
			kwargs['force'] = False
		
		key = os.path.basename (file)
		
		if kwargs['force'] or not self.is_file_exists (bucket, key):
			
			if not self.is_bucket_exists (bucket):
				self.create_bucket (bucket)
			
			if 'callback' not in kwargs:
				kwargs['callback'] = S3Progress (key, os.path.getsize (file))
			
			self.s3_client.upload_file (file, bucket, key, Callback = kwargs['callback'])
			
		mime_type, _ = mimetypes.guess_type (file)
		
		return self.generate_get_url (bucket, key, mime_type, **kwargs)
	
	def upload_bytes (self, bytes: bytes, bucket: str, key: str, mime_type: str, **kwargs) -> str:
		
		if not self.is_bucket_exists (bucket):
			self.create_bucket (bucket)
		
		self.s3_client.put_object (
			Bucket = bucket,
			Key = key,
			Body = bytes,
			ContentType = mime_type,
		)
		
		return self.generate_get_url (bucket, key, mime_type, **kwargs)
		
	def is_bucket_exists (self, bucket: str) -> bool:
		
		try:
			
			self.s3_client.head_bucket (Bucket = bucket)
			return True
		
		except ClientError as e:
			
			error_code = e.response['Error']['Code']
			
			if error_code == '404':  # 404 means the file does not exist in the entire global AWS ecosystem
				return False
			elif error_code == '403':  # 403 means the file exists globally, but you don't have permission to access it
				raise RuntimeError (f'Error: Bucket "{bucket}" already exists but owned by another account.')
			else:
				raise RuntimeError (e)
		
	def get_file_info (self, bucket: str, file: str):
		return self.s3_client.head_object (Bucket = bucket, Key = file)
	
	def is_file_exists (self, bucket: str, file: str) -> bool:
		
		try:
			
			self.get_file_info (bucket, file)
			return True
			
		except ClientError as e:
			
			error_code = e.response['Error']['Code']
			
			if error_code == '404':  # 404 means the file does not exist in the entire global AWS ecosystem
				return False
			elif error_code == '403':  # 403 means the file exists globally, but you don't have permission to access it
				raise RuntimeError (f'Error: File "{file}" already exists but owned by another account.')
			else:
				raise RuntimeError (e)
		
	def create_bucket (self, bucket: str, region: str = 'us-east-1'):
		
		if region == 'us-east-1':  # us-east-1 (N. Virginia) cannot have a LocationConstraint configuration passed
			self.s3_client.create_bucket (Bucket = bucket)
		else:
			self.s3_client.create_bucket (
				Bucket = bucket,
				CreateBucketConfiguration = { 'LocationConstraint': region }
			)
		
	def download_file (self, file, bucket: str, key: str, **kwargs):
		
		config = TransferConfig (
			use_threads = True,
			max_concurrency = self.num_threads,
			multipart_threshold = 1024 * 1024 * 25,  # 25 Mb
			multipart_chunksize = 1024 * 1024 * 25,
		)
		
		self.s3_client.download_file (
			bucket,
			key,
			file,
			Callback = kwargs['callback'],
			Config = config,
		)
		
class S3Progress (object):
	
	def __init__ (self, filename, file_size):
		
		self._filename = filename
		self._size = file_size
		self._length = 0
		self._lock = threading.Lock ()
		
	def __call__ (self, bytes_amount):
		
		with self._lock:
			
			self._length += bytes_amount
			
			percentage = (self._length / self._size) * 100
			
			sys.stdout.write (
				"\r%s  %s / %s  (%.2f%%)" % (
					self._filename, self._length, self._size, percentage
				)
			)
			
			sys.stdout.flush ()