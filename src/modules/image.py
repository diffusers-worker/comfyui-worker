import base64
import hashlib
from io import BytesIO
import json
import os
from typing import LiteralString

import PIL
from PIL.Image import Image
import piexif

def load_image (path) -> Image:
	
	input_image = PIL.Image.open (path)
	
	if input_image.mode != 'RGB':
		input_image.convert ('RGB')
	
	return input_image

def save_image (image: Image, path, filename):
	
	if not os.path.isdir (path):
		os.makedirs (path)
	
	file = os.path.join (path, filename)
	
	image.save (file)
	
	return file

def get_image_bytes (image: Image, format: str, **kwargs) -> bytes:
	
	byte_io = BytesIO ()
	
	image.save (byte_io, format, **kwargs)
	
	return byte_io.getvalue ()

def encode_base64_image (image: Image, format: str, **kwargs) -> str:
	return base64_encode (get_image_bytes (image, format, **kwargs))

def decode_base64_image (base64_string, path, filename) -> LiteralString | str | bytes:
	return save_base64_image (base64_decode (base64_string), path, filename)

def base64_encode (image: bytes) -> str:
	return base64.b64encode (image).decode ('utf-8')

def base64_decode (image: bytes) -> bytes:
	return base64.b64decode (image)

def save_base64_image (decoded_bytes: bytes, path, filename) -> LiteralString | str | bytes:
	
	if not os.path.isdir (path):
		os.makedirs (path)
	
	file = os.path.join (path, filename)
	
	with open (file, 'wb') as f:
		f.write (decoded_bytes)
	
	return file

def load_image_metadata (image: Image) -> dict:
	
	if image.format == 'JPEG':
		
		if 'exif' in image.info:
			
			exif_dict = piexif.load (image.info['exif'])
			user_comment = exif_dict.get ('Exif', {}).get (piexif.ExifIFD.UserComment)
			
			if user_comment:
				
				# Cut first 8 bytes charset prefix ('ASCII\x00\x00\x00') and decode string
				json_string = user_comment[8:].decode ('utf-8')
				
				return json.loads (json_string)
				
			else:
				return {}
			
		else:
			return {}
		
	elif image.format == 'PNG':
		
		if 'parameters' in image.info:
			return json.loads (image.info['parameters'])
		else:
			return {}
		
	else:
		raise ValueError ('Invalid image format')

def prepare_exif (json_string: str)-> dict:
	
	# EXIF requires strictly bytes. Add the encoding signature 'ASCII\x00\x00\x00' before the text
	user_comment_bytes = b'ASCII\x00\x00\x00' + json_string.encode ('utf-8')
	
	# Create an EXIF data structure (the UserComment tag lives in the 'Exif' group)
	return { '0th': {}, 'Exif': { piexif.ExifIFD.UserComment: user_comment_bytes, }, 'GPS': {}, '1st': {}, }

def get_image_sha256 (image: Image, format: str, **kwargs) -> str:
	return get_bytes_sha256 (get_image_bytes (image, format, **kwargs))

def get_bytes_sha256 (bytes: bytes) -> str:
	return hashlib.sha256 (bytes).hexdigest ()

def stream_bytes_and_sha256 (file_path, chunk_size = 65536):
	
	sha256_hash = hashlib.sha256 ()
	
	with open (file_path, 'rb') as f:
		
		while chunk := f.read (chunk_size):
			
			sha256_hash.update (chunk)
			
			yield chunk