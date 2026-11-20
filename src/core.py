from contextlib import contextmanager
from datetime import datetime
import os
import random
import re

from modules.s3_client import S3Client

class Core:
	
	config: dict = {
		
		'device': 'cuda:0',
		'debug': 0,
		'verbose': False,
		'metadata': True,
		'results': 'base64',
		'cache_dir': '',
		'wildcards_path': '',
		'output_name': '%Y%m%d_%H%M%S_%f',
		'format': 'png',
		'downloader': {
			'huggingface': {'token': ''},
		},
		
	}
	
	s3_client: S3Client
	
	def init (self, **config):
		
		for key, value in self.config.items ():
			self.config[key] = config.get (key, value)
		
		self.s3_client = S3Client (self.config['downloader']['s3'])
		
	def random_line (self, file: str) -> str:
		
		file = open (file, 'rb')
		
		text = []
		
		for line in file:
			text.append (line.strip ())
		
		file.close ()
		
		return text[random.randrange (0, len (text))].decode ('utf-8')
	
	def file_date_name (self) -> str:
		
		date = datetime.today ()
		
		pat = re.compile (r'%(.+?)')
		
		pos = 0
		
		name = self.config['output_name']
		
		while m := pat.search (name, pos):
			
			pos = m.start () + 1
			name = name.replace (m[0], date.strftime (m[0]))
		
		return name
	
	def progress (self, step: int, timestep: int, latents):
		print (step, timestep, latents[0][0][0][0])
	
	def verbose (self, mess: str):
		
		if self.config['verbose']:
			print (mess)
	
@contextmanager
def suppress_stderr ():
	
	"""Temporarily redirects stderr (C++ errors) to the void."""
	
	# Open the system "black hole" (for Windows this is 'nul', for Linux/Mac - '/dev/null')
	null_fds = [os.open (os.devnull, os.O_WRONLY)]
	
	# Save the original stderr
	save_fds = [os.dup (2)]
	
	try:
		
		# Redirect stderr (handle 2) to null
		os.dup2 (null_fds[0], 2)
		yield
	
	finally:
		
		# Return everything back
		os.dup2 (save_fds[0], 2)
		
		os.close (null_fds[0])
		os.close (save_fds[0])