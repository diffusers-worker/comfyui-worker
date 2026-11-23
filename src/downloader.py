import os
import re
import requests

from huggingface_hub import snapshot_download
from urllib.parse import urlparse, parse_qsl, unquote
from typing import List, Optional, Callable, LiteralString
from concurrent.futures import ThreadPoolExecutor

from requests import RequestException

from modules.s3_client import S3Client, S3Progress

class Downloader:
	
	"""
	An OOP-driven Facade that encapsulates pipeline weight resolution,
	cross-model byte-level progress aggregation, and atomic persistent storage delivery.
	"""
	
	config: dict
	download_callback: Optional[Callable[[int], None]]
	
	# Cumulative progress state variables
	accumulated_downloaded_bytes: int = 0
	current_file_previously_downloaded_bytes: int = 0
	
	chunk_size = 4 * 1024 * 1024
	num_threads = 8
	
	_s3_client: S3Client
	
	def __init__ (self, config: dict):
		self.config = config
		
	def generate_local_path (self, source: str, **kwargs):
		
		"""
		Private-facing worker module wrapped inside an entity boundary.
		Downloads, caches, and syncs an individual asset while managing internal tracking metrics.
		"""
		
		# Reset internal tracker offset tracking variables for this specific file scope execution
		self.current_file_previously_downloaded_bytes = 0
		
		target_folder = os.path.join (self.config['workspace_dir'], str (kwargs['subfolder']))
		
		os.makedirs (target_folder, exist_ok = True)
		
		if re.match (r'^[a-zA-Z0-9\-_.]+/[a-zA-Z0-9\-_.]+$', source): # HuggingFace repository
			
			author, repo = source.split ('/')
			
			# snapshot_download downloads the raw folder layout configuration natively.
			# To make it report progress to our abstract custom callback, we can pass our update function hooks.
			# But since snapshot_download runs multiple internal threads, we simulate global progress state steps.
			final_local_path = snapshot_download (
				repo_id = source,
				repo_type = 'model',
				local_dir = os.path.join (target_folder, author, repo),  # Pull repository folders directly into this exact named token path node
				token = self.config['huggingface']['token']
			)
			
			# Simulate instant completion update metrics verification state triggers
			# if os.path.exists (final_local_path):
			#	  repo_size = sum (os.path.getsize (os.path.join (r, f)) for r, _, files in os.walk (final_local_path) for f in files)
			#	  self._update_progress (repo_size, repo_size)
			
		else: # URL or relative path
			
			parsed_url = urlparse (source)
			
			if parsed_url.scheme in ['http', 'https']:  # URL
				
				final_local_path = self.get_remote_url (parsed_url, target_folder, **kwargs)
				
			else:  # Relative local file
				
				filename = os.path.basename (source)
				
				final_local_path = os.path.join (target_folder, filename)
				
				# Local cache check
				if os.path.exists (final_local_path):
					
					file_size = os.path.getsize (final_local_path)
					
					if 'callback' not in kwargs:
						kwargs['callback'] = S3Progress (filename, file_size)
					
					# If present, immediately simulate completing this specific file weight allocation
					self._update_progress (kwargs['callback'], file_size)
					
				elif os.path.exists (source):
					final_local_path = source
				else:
					raise RuntimeError (f'File {final_local_path} not found')
				
		return final_local_path
		
	def download_chunk_sync (self, url, domain, callback, start, end, file):
		
		headers = { 'Range': f'bytes={start}-{end}' }
		
		if 'civitai.com' in domain or 'civitai.red' in domain:
			headers['Authorization'] = f'Bearer {self.config["tokens"]["civitai"]}'
		
		# stream = True allows to write to a file in chunks without filling up the worker RAM
		with requests.get (url, headers = headers, stream = True, timeout = 60) as r:
			
			r.raise_for_status ()
			
			with open (file, 'r+b') as f:
				
				f.seek (start)
				
				for chunk in r.iter_content (chunk_size = self.chunk_size):
					
					f.write (chunk)
					
					self._update_progress (callback, f.tell ())
		
	def get_remote_url (self, parsed_url, target_folder, **kwargs) -> LiteralString | str | bytes:
		
		url = parsed_url.geturl ()
		domain = parsed_url.netloc.lower ()
		
		if 'civitai.com' in domain or 'civitai.red' in domain:
			
			if '/models/' in parsed_url.path:
				
				model_id = parsed_url.path.split ('/models/')[-1].split ('/')
				url = f'https://civitai.com{model_id}'
			
			else:
				model_id = 'unknown'
			
			if self.config['tokens']['civitai']:
				
				connector = '&' if '?' in url else '?'
				url += f'{connector}token={self.config["tokens"]["civitai"]}'
				
			#filename = f'civitai_{model_id}.safetensors'
			
		with requests.head (url, allow_redirects = True) as response:
			
			filename = None
			
			if 'content-disposition' in response.headers:
				
				cd = response.headers['content-disposition']
				names = re.findall (r'filename=(.+)', cd)
				
				if names:
					filename = names[0].strip ('"')
				
				file_size = int (response.headers.get ('content-length', 0))
			
			if not filename:
				
				params = dict (parse_qsl (parsed_url.query))
				
				disposition = params.get ('response-content-disposition', '')
				
				if "filename*=UTF-8''" in disposition:
					
					raw_name = disposition.split ("filename*=UTF-8''")[-1]
					filename = unquote (raw_name)
				
				if not filename:
					
					# На случай если HEAD не поддерживается сервером
					response = requests.get (url, stream = True)
					
					if 'content-disposition' in response.headers:
						
						cd = response.headers['content-disposition']
						names = re.findall (r'filename=(.+)', cd)
						
						if names:
							filename = names[0].strip ('"')
						
						file_size = int (response.headers.get ('content-length', 0))
					
			if filename:
				
				final_local_path = os.path.join (target_folder, filename)
				temp_local_path = f'{final_local_path}.tmp'
				
				exists = os.path.exists (final_local_path)
				
				if 'force' not in kwargs:
					kwargs['force'] = False
				
				# Local cache check
				if kwargs['force'] or not exists:
					
					if 'callback' not in kwargs:
						kwargs['callback'] = S3Progress (filename, file_size)
					
					if 'civitai.com' in domain or 'civitai.red' in domain:
						
						try:
							
							# 2. Создаем размеченный файл
							with open (temp_local_path, 'wb') as f:
								f.truncate (file_size)
							
							chunk_size = file_size // self.num_threads
							
							# 3. Разгоняем пул потоков
							with ThreadPoolExecutor (max_workers = self.num_threads) as executor:
								
								futures = []
								
								for i in range (self.num_threads):
									
									start = i * chunk_size
									end = file_size - 1 if i == self.num_threads - 1 else (start + chunk_size - 1)
									
									# Отправляем задачу в пул
									futures.append (executor.submit (
										self.download_chunk_sync,
										url,
										domain,
										kwargs['callback'],
										start,
										end,
										temp_local_path
									))
									
								# Ждем выполнения всех потоков
								for future in futures:
									future.result ()
								
							if not exists:
								os.rename (temp_local_path, final_local_path)
							
						except Exception as e:
							
							if os.path.exists (temp_local_path):
								os.remove (temp_local_path)
							
							raise RuntimeError (e)
							
					else:
						
						self._s3_client.download_file (
							final_local_path,
							kwargs['subfolder'],
							os.path.basename (final_local_path),
							callback = kwargs['callback']
						)
						
				return final_local_path
				
		raise RuntimeError (f'Can\'t download {url}')
		
	def _update_progress (self, callback, file_downloaded_bytes: int):
		
		"""
		Encapsulated internal callback hook that increments global bytes metrics.
		"""
		
		new_bytes_chunk = file_downloaded_bytes - self.current_file_previously_downloaded_bytes
		
		self.current_file_previously_downloaded_bytes = file_downloaded_bytes
		self.accumulated_downloaded_bytes += new_bytes_chunk
		
		if callback:
			callback (self.accumulated_downloaded_bytes)
	
	def calculate_total_pipeline_weight (self, sources: List[str]) -> int:
		
		"""
		Performs rapid non-blocking HTTP HEAD checks to aggregate total pipeline bytes.
		"""
		
		total: int = 0
		
		for source in sources:
			
			if not source.startswith (('http://', 'https://')):
				
				total += self.chunk_size * 1024  # Fallback estimate (~4GB) for HF Repo IDs
				continue
				
			try:
				
				response = requests.head (
					source,
					timeout = 3,
					allow_redirects = True
				)
				
				if response.status_code == 200 and 'content-length' in response.headers:
					total += int (response.headers['content-length'])
				
			except RequestException:
				total += 2 * 1024 * 1024 * 1024  # Safe generic fallback allocation
		
		return total if total > 0 else 1