import math
import os
from pathlib import Path

from PIL import Image
from safetensors.numpy import load_file
import torch

from comfy_worflow import ComfyWorkflow
from core import Core
from custom_nodes.image_in_memory import ReturnImageInMemory

import nodes as root_nodes
from comfy_ui import nodes as pkg_nodes
import folder_paths as root_fp
from comfy_ui import folder_paths as pkg_fp

class Text2Image2 (ComfyWorkflow):
	
	routes = {
		
		'model': '',
		'checkpoint': '',
		'gguf': '',
		'vae': '',
		'scheduler': None,
		'loras': [],
		
	}
	
	pipe_kwargs = {}
	image_kwargs = {}
	
	seed: int = 0
	
	# Standard Aspect Ratio buckets optimized for ~1 Megapixel resolution (SDXL, Flux, SD3, Qwen Image)
	
	BUCKETS_1_MP = {
		
		'1:1': (1024, 1024),
		'16:9': (1344, 768),
		'9:16': (768, 1344),
		'3:2': (1216, 832),
		'2:3': (832, 1216),
		'4:3': (1152, 896),
		'3:4': (896, 1152),
		'21:9': (1536, 640),
		'9:21': (640, 1536),
		
	}
	
	# Standard Aspect Ratio buckets optimized for ~0.25 Megapixel resolution (Legacy SD 1.5)
	
	BUCKETS_05_MP = {
		
		'1:1': (512, 512),
		'16:9': (768, 448),
		'9:16': (448, 768),
		'3:2': (640, 448),
		'2:3': (448, 640),
		'4:3': (640, 512),
		'3:4': (512, 640),
		
	}
	
	def __init__ (self, core: Core):
		super ().__init__ (core)
	
	def init (self, data: dict):
		
		self.data.update ({
			
			'model': '',
			'checkpoint': '',
			
			'loras': [],
			
			'prompt': '',
			'negative_prompt': '',
			
			'sampler': 'euler_ancestral',
			'scheduler': 'normal',
			'steps': 30,
			'seed': 0,
			'cfg_scale': 4.0,
			'cfg_rescale': 0.0,
			'clip_skip': 2,
			'images_per_prompt': 1,
			'freeu': {},
			'width': 0,
			'height': 0,
			'ratio': '2:3',
			
			'vae': '',
			'gguf': '',
			'adetailer': {},
			'controlnet': {},
			'ip_adapter': {},
			
		})
		
		super ().init (self.data)
		
		if not self.data['model'] and not self.data['checkpoint']:
			raise ValueError ('No model or checkpoint provided')
		
		if not self.data['prompt']:
			raise ValueError ('No prompt provided')
		
		for fp in (root_fp, pkg_fp):
			for folder in ["models", "checkpoints", "loras"]:
				fp.add_model_folder_path (folder, os.path.join (self.core.config['downloader']['workspace_dir'], folder))
		
		for mod in (root_nodes, pkg_nodes):
			mod.NODE_CLASS_MAPPINGS["ReturnImageInMemory"] = ReturnImageInMemory
		
		self.switches['model'] = False
		self.switches['checkpoint'] = False
		
		for value in ['model', 'checkpoint', 'loras', 'scheduler', 'gguf']:
			
			self.switches[value] = self.routes[value] != self.data[value]
			
			if self.switches[value]:
				
				if self.data['model']:
					self.switches['model'] = True
				else:
					self.switches['checkpoint'] = True
		
	def build_graph (self, data: dict) -> str:
		
		self.weight_index = 0
		
		self.process_pipeline ()
		
		# Hot-swap the transformer block if the GGUF file has changed
		self.manage_gguf ()
		
		# Handle VAE adjustments (First load or Hot-swap are processed identically here)
		self.manage_vae ()
		
		if self.data['model']:
			
			self.reload_base_model ()
			
			file = self.generate_local_path (self.data['model'], subfolder = 'models')
			
			self.image_metadata['model'] = os.path.basename (file)
			
			self.pipe = self.model['pipeline'].from_pretrained (
				file,
				**self.pipe_kwargs,
			)
			
			# ONNX Runtime # TODO
			if 'optimum.intel' in type (self.pipe).__module__:
				
				self.pipe = self.model['pipeline'].from_pretrained (
					self.generate_local_path (self.data['model'], subfolder = 'models'),
					**self.pipe_kwargs,
				)
			
			self.routes['model'] = self.data['model']
			
		elif self.data['checkpoint']:
			
			self.reload_base_model ()
			
			file = self.generate_local_path (self.data['checkpoint'], subfolder = 'checkpoints')
			
			self.image_metadata['checkpoint'] = os.path.basename (file)
			
			# 1. Basic checkpoint (we initialize three main threads here: model, clip, vae)
			ckpt = self.g.add ("CheckpointLoaderSimple", ckpt_name = self.image_metadata['checkpoint'])
			
			self.image_kwargs['model'] = ckpt.out (0)
			self.image_kwargs['clip'] = ckpt.out (1)
			self.image_kwargs['vae'] = ckpt.out (2)
			
			self.routes['checkpoint'] = self.data['checkpoint']
			
		# self.core.verbose (f'{self.pipe.__class__.__name__} loaded')
		
		# Handle active multi-adapter LoRA weights synchronization
		self.manage_loras ()
		
		self.optimize ()
		
	images: list[Image.Image] = []
	
	@torch.inference_mode ()
	def get_pipeline (self, data: dict, **model):
		
		self.model = model
		
		self.init (self.data)
		
		self.core.verbose ('Start generation...')
		
		# 3. Dynamic CLIP Skip: if > 1, skip the clip through CLIPSetLastLayer
		
		if self.data['clip_skip'] > 1:
			
			clip_layer = self.g.add ("CLIPSetLastLayer", stop_at_clip_layer = -self.data['clip_skip'], clip = self.image_kwargs['clip'])
			self.image_kwargs['clip'] = clip_layer.out (0)
			
		# 4. Prompts (get the final clip after clip_skip and all LoRA)
		
		pos = self.g.add (
			"CLIPTextEncode",
			text = self.image_kwargs['prompt'],
			clip = self.image_kwargs['clip'],
		)
		
		neg = self.g.add (
			"CLIPTextEncode",
			text = self.image_kwargs['negative_prompt'],
			clip = self.image_kwargs['clip'],
		)
		
		# 6. Latent: branching txt2img vs img2img based on the presence of an input image
		if "image" in self.data:
			
			load_img = self.g.add ("LoadImage", image = self.data["image"])
			latent_node = self.g.add ("VAEEncode", pixels = load_img.out (0), vae = self.image_kwargs['vae'])
			denoise = float (self.data.get ("denoise", 0.75))
			
		else:
			
			latent_node = self.g.add (
				"EmptyLatentImage",
				width = self.image_kwargs["width"],
				height = self.image_kwargs["height"],
				batch_size = self.data['images_per_prompt']
			)
			
			denoise = 1.0
			
		# Apply IP-Adapter if a reference image is provided in the request payload
		if "image" in self.data["ip_adapter"]:
			
			# 1. Decode reference image from base64 in memory
			ip_ref = self.g.add ("LoadImageFromBase64", base64_str = self.data["ip_adapter"]["image"])
			
			# 2. Load unified IP-Adapter and CLIP Vision models via preset
			# Supported presets: "LIGHT - SD1.5 only (low strength)", "STANDARD (medium strength)",
			# "PLUS (high strength)", "PLUS FACE (portraits)", "FULL FACE - SD1.5 only (portraits stronger)"
			ip_loader = self.g.add (
				"IPAdapterUnifiedLoader",
				model = self.image_kwargs['model'],
				preset = self.data["ip_adapter"].get ("preset", "PLUS (high strength)")
			)
			
			# 3. Apply IP-Adapter conditioning to the model pipeline
			ip_apply = self.g.add (
				"IPAdapterAdvanced",
				model = ip_loader.out (0),
				ipadapter = ip_loader.out (1),
				image = ip_ref.out (0),
				weight = float (self.data["ip_adapter"].get ("weight", 0.75)),
				weight_type = self.data["ip_adapter"].get ("weight_type", "linear"),
				combine_embeds = self.data["ip_adapter"].get ("combine_embeds", "concat"),
				start_at = float (self.data["ip_adapter"].get ("start", 0.0)),
				end_at = float (self.data["ip_adapter"].get ("end", 1.0)),
				embeds_scaling = self.data["ip_adapter"].get ("embeds_scaling", "V only")
			)
			
			# Route the updated model output further down the pipeline
			self.image_kwargs['model'] = ip_apply.out (0)
			
		sampler = self.g.add (
			"KSampler",
			seed = int (self.data["seed"]),
			steps = int (self.data["steps"]),
			cfg = float (self.data["cfg_scale"]),
			sampler_name = self.data["sampler"],
			scheduler = self.data["scheduler"],
			denoise = denoise,
			model = self.image_kwargs['model'],
			positive = pos.out (0),
			negative = neg.out (0),
			latent_image = latent_node.out (0)
		)
		
		decode = self.g.add (
			"VAEDecode",
			samples = sampler.out (0),
			vae = self.image_kwargs['vae'],
		)
		
		image = decode.out (0)
		
		# Ветка DWPose ControlNet (активируется, если в JSON передана картинка позы)
		if "dwpose" in self.data:
			
			# 1. Загружаем референс позы из base64
			pose_ref = self.g.add ("LoadImageFromBase64", base64_str = self.data['dwpose'].get ("image"))
			
			# 2. Извлекаем скелет через DWPose
			dwpose = self.g.add (
				"DWPreprocessor",
				detect_hand = "enable",
				detect_body = "enable",
				detect_face = "enable",
				resolution = self.data["width"],
				bbox_detector = self.data['dwpose'].get ("bbox_detector", "yolox_l.onnx"),
				pose_estimator = self.data['dwpose'].get ("pose_estimator", "dw-ll_ucoco_384_bs5.onnx"),
				scale_stick_for_xinsr_cn = self.data['dwpose'].get ("scale_stick", "disable"),
				image = pose_ref.out (0)
			)
			
			# 3. Загружаем модель ControlNet OpenPose для SDXL / Illustrious
			cn_loader = self.g.add (
				"ControlNetLoader",
				control_net_name = self.data['dwpose'].get ("model", "controlnet-openpose-sdxl.safetensors")
			)
			
			# 4. Применяем ControlNet к positive и negative кондишенингам
			cn_apply = self.g.add (
				"ControlNetApplyAdvanced",
				strength = float (self.data['dwpose'].get ("strength", 0.85)),
				start_percent = float (self.data['dwpose'].get ("start", 0.0)),
				end_percent = float (self.data['dwpose'].get ("end", 1.0)),
				positive = pos.out (0),
				negative = neg.out (0),
				control_net = cn_loader.out (0),
				image = dwpose.out (0),
				vae = self.image_kwargs['vae']
			)
			
			# Перенаправляем потоки positive и negative на выход ControlNet
			pos_out = cn_apply.out (0)
			neg_out = cn_apply.out (1)
			
		else:
			
			pos_out = pos.out (0)
			neg_out = neg.out (0)
			
		# Ветка FaceDetailer (активируется флагом "use_face_detailer": true)
		if "face_detailer" in self.data:
			
			# Загружаем детектор лиц (стандартный face_yolov8m.pt из Impact Pack)
			bbox_det = self.g.add (
				"UltralyticsDetectorProvider",
				model_name = self.data["face_detailer"].get ("model", "bbox/face_yolov8m.pt"),
			)
			
			detailer = self.g.add (
				"FaceDetailer",
				image = image,
				model = self.image_kwargs['model'],  # Подаём модель уже со всеми вашими LoRA!
				clip = self.image_kwargs['clip'],
				vae = self.image_kwargs['vae'],
				positive = pos_out,
				negative = neg_out,
				bbox_detector = bbox_det.out (0),
				guide_size = float (self.data["face_detailer"].get ("guide_size", 512)),
				guide_size_for = True,  # True = ориентироваться по bbox лица
				max_size = 1024.0,
				seed = int (self.data["seed"]) + 1,
				steps = int (self.data["face_detailer"].get ("steps", 20)),
				cfg = float (self.data["cfg_scale"]),
				sampler_name = self.data["sampler"],
				scheduler = self.data["scheduler"],
				denoise = float (self.data["face_detailer"].get ("denoise", 0.4)),
				feather = self.data["face_detailer"].get ("feather", 5),
				noise_mask = True,
				force_inpaint = True,
				bbox_threshold = self.data["face_detailer"].get ("bbox_threshold", 0.5),
				bbox_dilation = self.data["face_detailer"].get ("bbox_dilation", 10),
				bbox_crop_factor = self.data["face_detailer"].get ("bbox_crop_factor", 3.0),
				sam_detection_hint = self.data["face_detailer"].get ("sam_detection_hint", "center-1"),
				sam_dilation = self.data["face_detailer"].get ("sam_dilation", 0),
				sam_threshold = self.data["face_detailer"].get ("sam_threshold", 0.93),
				sam_bbox_expansion = self.data["face_detailer"].get ("sam_bbox_expansion", 0),
				sam_mask_hint_threshold = self.data["face_detailer"].get ("sam_mask_hint_threshold", 0.7),
				sam_mask_hint_use_negative = self.data["face_detailer"].get ("sam_mask_hint_use_negative", False),
				drop_size = self.data["face_detailer"].get ("drop_size", 10),
				wildcard = self.data["face_detailer"].get ("wildcard", ""),
				cycle = self.data["face_detailer"].get ("cycle", 1)
			)
			
			# Перенаправляем финальный поток картинки на выход FaceDetailer
			image = detailer.out (0)
			
		out_node = self.g.add ("ReturnImageInMemory", images = image)
		
		self.images = self.render (self.g.nodes, out_node.id)
		
		return self
	
	def process_pipeline (self):
		
		self.switches['model'] = False
		self.switches['checkpoint'] = False
		
		self.pipe_kwargs = self.model_kwargs ()
		
		self.image_kwargs = {
			
			'prompt': self.process_prompt (self.data['prompt']),
			'negative_prompt': self.process_prompt (self.data['negative_prompt']),
			
		}
		
		self.calculate_dimensions ()
		
		self.image_metadata = {
			
			'workflow': 'text2image',
			'prompt': self.image_kwargs['prompt'],
			'negative_prompt': self.image_kwargs['negative_prompt'],
			'sampler': self.data['sampler'],
			'cfg_scale': self.data['cfg_scale'],
			'cfg_rescale': self.data['cfg_rescale'],
			'clip_skip': self.data['clip_skip'],
			'steps': self.data['steps'],
			'seed': self.data['seed'],
			'loras': [],
			
		}
		
	def manage_gguf (self):
		
		if self.data['gguf']:
			
			model_kwargs = self.model_kwargs ()
			
			model_kwargs['quantization_config'] = GGUFQuantizationConfig (
				compute_dtype = self.model['dtype']
			)
			
			model_kwargs['config'] = self.data['gguf']
			
			self.pipe_kwargs['transformer'] = self.model['transformer'].from_single_file (
				self.generate_local_path (self.data['gguf'], subfolder = 'gguf'),
				**model_kwargs,
			)
			
			self.routes['gguf'] = self.data['gguf']
			
			self.reload_model ()
			
	def manage_vae (self):
		
		"""
		Handles both the initial injection and the on-the-fly hot swapping of the VAE module.
		Ensures the operation is done in-place without reloading heavy text encoders.
		"""
		
		if self.data['vae']:
			
			vae_node = self.g.add ("VAELoader", vae_name = self.data['vae'])
			
			self.image_kwargs['vae'] = vae_node.out (0)
			
			self.routes['vae'] = self.data['vae']
			
	def manage_loras (self):
		
		if self.data['loras']:
			
			self.image_metadata['loras'] = []
			
			for idx, lora in enumerate (self.data["loras"]):
				self.load_lora_weights (lora, f'adapter_{idx}')
			
			self.routes['loras'] = self.data['loras']
			
	def load_lora_weights (self, lora: dict, adapter_name):
		
		file = self.generate_local_path (lora['source'], subfolder = 'loras')
		
		if 'weights' in lora:
			
			modified_state_dict = {}
			multiplier_mask = lora['weights']
			
			# Iterate through weights and apply multipliers based on SDXL UNet architecture
			for key, tensor in load_file (file).items ():
				
				weight = 1.0  # Default multiplier for text encoders
				
				if 'unet' in key:
					
					if 'mid_block' in key:
						
						# Key geometry layer (eyes, mouth structure, face shapes)
						weight = multiplier_mask.get ('mid', 1.8)
						
					elif 'down_blocks' in key:
						
						# Input blocks responsible for major composition and layout
						block_idx = int (key.split ('down_blocks.')[1].split ('.')[0])
						weight = multiplier_mask.get (f'down_{block_idx}', 1.2)
						
					elif 'up_blocks' in key:
						
						# Output blocks responsible for final textures, styling, and lineart
						block_idx = int (key.split ('up_blocks.')[1].split ('.')[0])
						weight = multiplier_mask.get (f'up_{block_idx}', 0.1)
					
				# Modify the tensor weights in memory
				modified_state_dict[key] = tensor * weight
				
			# Inject modified weights into the diffusers pipeline using an adapter name
			self.pipe.load_lora_weights (modified_state_dict, adapter_name = adapter_name)
			
		else:
			
			weight = float (lora.get ("weight", 1.0))
			
			lora_node = self.g.add (
				"LoraLoader",  # Or DynamicLoraLoaderFromURL
				model = self.image_kwargs['model'],
				clip = self.image_kwargs['clip'],
				lora_name = os.path.basename (file),
				strength_model = weight,
				strength_clip = float (lora.get ("clip_weight", weight))
			)
			
			self.image_kwargs['model'] = lora_node.out (0)
			self.image_kwargs['clip'] = lora_node.out (1)
			
		new_lora = {}
		
		for key, value in lora.items ():
			
			if key == 'source':
				new_lora['name'] = Path (file).stem
			else:
				new_lora[key] = value
			
		self.image_metadata['loras'].append (new_lora)
	
	def reload_base_model (self):
		
		# Reset VAE and loras tracker because a new base model brings its own default internal VAE and loras #TODO
		self.routes['vae'] = ''
		self.routes['loras'] = []
	
	def calculate_dimensions (self):
		
		if self.data['width'] or self.data['height']:
			
			if 'default_dim' not in self.model:
				self.model['default_dim'] = 1024
			
			if 'div_factor' not in self.model:
				self.model['div_factor'] = 16
			
			if self.data['width']:
				self.image_kwargs['width'] = math.ceil (self.data['width'] / self.model['div_factor']) * self.model['div_factor']
			else:
				self.image_kwargs['width'] = self.model['default_dim']
			
			if self.data['height']:
				self.image_kwargs['height'] = math.ceil (self.data['height'] / self.model['div_factor']) * self.model['div_factor']
			else:
				self.image_kwargs['height'] = self.model['default_dim']
			
			return
		
		if 'buckets' not in self.model:
			self.model['buckets'] = self.BUCKETS_1_MP
		
		if self.data['ratio'] in self.model['buckets']:
			
			sizes = self.model['buckets'][self.data['ratio']]
			
			self.image_kwargs['width'] = sizes[0]
			self.image_kwargs['height'] = sizes[1]
			
		else:
			raise ValueError (f'Wrong ratio {self.data["ratio"]}')