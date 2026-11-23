from abc import ABC
import gc
import json
import os
import random
import re
from typing import List

from PIL import PngImagePlugin
import torch

from core import Core
from PIL.Image import Image
from downloader import Downloader
from modules.image import prepare_exif, get_image_bytes, base64_encode, get_bytes_sha256

class Workflow (ABC):
	
	core: Core
	
	data: dict = {}
	model: dict = {}
	switches: dict = {}
	
	device: List[str] = []
	
	pipe = None
	
	image_metadata: dict = {}
	
	downloader: Downloader
	
	def __init__ (self, core: Core):
		
		self.core = core
		
		self.downloader = Downloader (self.core.config['downloader'])
		
		self.downloader._s3_client = self.core.s3_client
		
	def init (self, data: dict):
		
		self.data.update (data)
		
		self.device = self.core.config['device'].split (':')
		
		if self.device[0] == 'cuda':
			
			if torch.cuda.is_available ():
				
				self.device_id = torch.cuda.current_device ()
				
				self.core.verbose (f'CUDA found on device {self.device_id}')
				
			else:
				raise RuntimeError ('CUDA is set but not available')
			
	def save_image (self, image: Image) -> str:
		
		self.image_metadata['software'] = 'Custom Serverless Diffusers Engine'
		
		json_string = json.dumps (self.image_metadata, ensure_ascii = False, indent = 2)
		
		if self.core.config['format'] == 'png':
			
			if self.core.config['metadata']:
				
				png_info = PngImagePlugin.PngInfo ()
				
				png_info.add_text ('parameters', json_string)
				
			else:
				png_info = None
			
			bytes = get_image_bytes (image, 'PNG', pnginfo = png_info)
			
			if self.core.config['results'] == 's3':
				
				sha256 = get_bytes_sha256 (bytes)
				mime_type = f'image/{self.core.config['format'].lower ()}'
				
				output = self.core.s3_client.upload_bytes (bytes, 'results', f'{sha256}.{self.core.config['format']}', mime_type)
				
			else:
				output = base64_encode (bytes)
			
			return output
			
		elif self.core.config['format'] == 'jpg':
			
			bytes = get_image_bytes (image, 'JPEG', quality = 95, exif = prepare_exif (json_string))
			
			if self.core.config['results'] == 's3':
				
				sha256 = get_bytes_sha256 (bytes)
				mime_type = f'image/{self.core.config['format'].lower ()}'
				
				output = self.core.s3_client.upload_bytes (bytes, 'results', f'{sha256}.{self.core.config['format']}', mime_type)
				
			else:
				output = base64_encode (bytes)
			
			return output
			
		else:
			raise RuntimeError (f'Wrong format {self.core.config["format"]}')
	
	def generate_local_path (self, source: str, **kwargs):
		
		if not self.core.config['verbose']:
			kwargs['callback'] = None
		
		return self.downloader.generate_local_path (source, **kwargs)
	
	def model_kwargs (self) -> dict:
		
		model_kwargs = {
			
			'use_safetensors': True,
			'low_cpu_mem_usage': True,
			'token': self.core.config['downloader']['huggingface']['token'],
			
		}
		
		if 'cache_dir' in self.core.config and self.core.config['cache_dir']:
			model_kwargs['ov_config'] = { 'CACHE_DIR': self.core.config['cache_dir'], }
		
		return model_kwargs
	
	def get_prompts (self, prompt_file: str) -> str:
		
		self.core.verbose ('Reading prompts file: ' + prompt_file)
		
		file = open (prompt_file, 'rb')
		
		return file[0].strip ().decode ('utf-8')
	
	def process_prompt (self, prompt) -> str:
		
		# Wildcards
		
		pat = re.compile (r'__(.+?)__')
		
		pos = 0
		
		while m := pat.search (prompt, pos):
			
			pos = m.start () + 1
			
			file = os.path.abspath (os.path.join (self.core.config['wildcards_path'], m[1] + '.txt'))
			
			if os.path.exists (file):
				
				line = self.core.random_line (file)
				prompt = prompt.replace (m[0], line, 1)
			
			else:
				raise ValueError ('Wildcard file ' + file + ' not found')
		
		# OR clause
		
		pattern = re.compile (r'{(.+?)}')
		
		pos = 0
		
		while m := pattern.search (prompt, pos):
			
			pos = m.start () + 1
			
			text = str (m[1]).split ('|')
			
			line = text[random.randrange (0, len (text))]
			
			prompt = self.process_prompt (prompt.replace (m[0], line, 1))
		
		return prompt
	
	# Optimization
	
	available_vram: float = 0
	effective_budget: float = 0
	weight_index: float = 0
	device_id = -1
	
	def get_free_vram_gb (self) -> float:
		
		"""
		Возвращает РЕАЛЬНО свободную память в данный момент, а не общую.
		"""
		
		if self.device_id == -1:
			return 0.0
		
		torch.cuda.empty_cache ()
		gc.collect ()
		
		total_mem = torch.cuda.get_device_properties (self.device_id).total_memory
		allocated_mem = torch.cuda.memory_allocated (self.device_id)
		
		return (total_mem - allocated_mem) / (1024 ** 3)
		
	def available_ram (self):
		
		# Получаем реально доступную чистую память перед генерацией
		self.available_vram = self.get_free_vram_gb ()
		
		# Выбираем стратегию на основе (Доступная Память минус Нагрузка Фичей)
		self.effective_budget = self.available_vram - (self.weight_index * 0.8)
		
		self.core.verbose (f'[MemManager] Доступно VRAM: {self.available_vram:.2f} GB. Индекс нагрузки: {self.weight_index}. Эффективный бюджет: {self.effective_budget:.2f} GB')
		
	def optimize (self):
		
		self.available_ram ()
		
		"""
		if 'controlnet' in features_enabled:
			self.weight_index += 3
		if 'ip_adapter' in features_enabled:
			self.weight_index += 2
		if 'adetailer' in features_enabled:
			self.weight_index += 2  # Отдельный проход инференса
		"""
		
		if self.device_id > -1:
			
			# 1. Базовый Dtype (Пропускаем для квантованных GGUF моделей)
			if not self.switches['gguf'] and self.pipe.dtype == torch.float32:
				
				# Проверяем, поддерживает ли карта bfloat16 (Ampere+ вроде RTX 30/40/50)
				
				if torch.cuda.is_bf16_supported ():
					self.pipe.to (self.model['dtype'], dtype = torch.bfloat16)
				else:
					self.pipe.to (self.model['dtype'], dtype = torch.float16)
				
			# 3. Применение стратегий
			
			if self.effective_budget < 4.0:  # КРИТИЧЕСКИЙ РЕЖИМ (Эффективный бюджет < 4ГБ)
				
				self.core.verbose ('-> Активация режима жесткой экономии (Sequential Offload)')
				
				self.pipe.enable_sequential_cpu_offload ()
			
			elif 4.0 <= self.effective_budget < 8.0:  # СРЕДНИЙ РЕЖИМ (Бюджет от 4 до 8ГБ)
				
				self.core.verbose ('-> Активация сбалансированного режима (Model Offload + VAE Slicing)')
				
				self.pipe.enable_model_cpu_offload ()
				
				# Включаем слайсинг только для VAE (самый частый виновник OOM при апскейле)
				
				if hasattr (self.pipe, 'vae'):
					
					self.pipe.vae.enable_slicing ()
					self.pipe.vae.enable_tiling ()  # Незаменимо для ADetailer при работе с большими лицами
			
			else:  # МАКСИМАЛЬНЫЙ РЕЖИМ (Бюджет > 8ГБ)
				
				self.core.verbose ('-> Активация режима максимальной производительности')
				
				self.pipe.to (self.model['dtype'])
				
				# Обязательно сбрасываем оффлоады, если они были включены ранее
				if hasattr (self.pipe, 'disable_model_cpu_offload'):
					
					try:
						self.pipe.disable_model_cpu_offload ()
					except:
						pass
			
			# 4. Встроенный SDPA vs XFormers
			
			# Если PyTorch свежий, он сам выберет лучший вариант внимания (SDPA), xFormers часто уже лишний.
			# Но для старых карт Pascal/Turing (GTX 10xx / RTX 20xx) xFormers все еще топ.
			
			major, _ = torch.cuda.get_device_capability (self.device_id)
			
			if major < 8:  # Карты старее RTX 3000
				
				try:
					self.pipe.enable_xformers_memory_efficient_attention ()
				except Exception:
					pass