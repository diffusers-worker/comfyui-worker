import math
import os
from pathlib import Path

from PIL.Image import Image

from comfy_worflow import ComfyWorkflow

import nodes as root_nodes
from comfy_ui import nodes as pkg_nodes
import folder_paths as root_fp
from comfy_ui import folder_paths as pkg_fp
from custom_nodes.return_image_list import ReturnImageList

class Text2Image (ComfyWorkflow):
	
	image_kwargs = {}
	
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
	
	def init (self, data: dict):
		
		super ().init (data)
		
		if not self.data['model'] and not self.data['checkpoint']:
			raise ValueError ('No model or checkpoint provided')
		
		if not self.data['prompt']:
			raise ValueError ('No prompt provided')
		
		self.image_kwargs = {
			
			'prompt': self.process_prompt (self.data['prompt']),
			'negative_prompt': self.process_prompt (self.data['negative_prompt']),
			
		}
		
		self.set_image_metadata ()
		
		for fp in (root_fp, pkg_fp):
			for folder in ["models", "checkpoints", "loras"]:
				fp.add_model_folder_path (folder, os.path.join (self.core.config['downloader']['workspace_dir'], folder))
		
	def set_image_metadata (self):
		
		self.image_kwargs = {
			
			'prompt': self.process_prompt (self.data['prompt']),
			'negative_prompt': self.process_prompt (self.data['negative_prompt']),
			
		}
		
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
			
		}
	
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
	
	def get_images (self, data: dict) -> list[Image]:
		
		self.data = {
			
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
			'controlnet': {},
			'ip_adapter': {},
			'face_detailer': {},
			
		}
		
		self.init (data)
		
		self.core.verbose ('Generating images...')
		
		for mod in (root_nodes, pkg_nodes):
			
			mod.NODE_CLASS_MAPPINGS.update ({
				"ReturnImageList": ReturnImageList,
			})
		
		model, clip, vae, pos, neg = self.load_base_checkpoint ()
		
		self.calculate_dimensions ()
		
		latent_node = self.g.add (
			"EmptyLatentImage",
			width = self.image_kwargs["width"],
			height = self.image_kwargs["height"],
			batch_size = self.data['images_per_prompt']
		)
		
		self.load_ksampler (model, clip, vae, latent_node.out (0), 1.0, pos, neg)
		
		return ReturnImageList.consume ()
	
	def load_base_checkpoint (self):
		
		"""
		Loads base checkpoint and optionally overrides VAE and CLIP skip.
		"""
		
		file = self.generate_local_path (self.data['checkpoint'], subfolder = 'checkpoints')
		
		self.image_metadata['checkpoint'] = Path (file).stem
		
		# 1. Basic checkpoint (we initialize three main threads here: model, clip, vae)
		ckpt = self.g.add (
			"CheckpointLoaderSimple",
			ckpt_name = os.path.basename (file)
		)
		
		model, clip, vae = ckpt.out (0), ckpt.out (1), ckpt.out (2)
		
		if self.data['vae']:
			
			file = self.generate_local_path (self.data['vae'], subfolder = 'vae')
			
			self.image_metadata['vae'] = Path (file).stem
			
			vae_node = self.g.add (
				"VAELoader",
				vae_name = os.path.basename (file)
			)
			
			vae = vae_node.out (0)
		
		if self.data['clip_skip'] > 1:
			
			clip_layer = self.g.add (
				"CLIPSetLastLayer",
				stop_at_clip_layer = -self.data['clip_skip'],
				clip = clip
			)
			
			clip = clip_layer.out (0)
		
		# 2. Apply LoRA chain and optional IP-Adapter
		model, clip = self.attach_loras (model, clip)
		model = self.attach_ip_adapter (model)
		
		# 3. Encode positive and negative prompts
		
		pos_node = self.g.add (
			"CLIPTextEncode",
			text = self.image_kwargs['prompt'],
			clip = clip,
		)
		
		neg_node = self.g.add (
			"CLIPTextEncode",
			text = self.image_kwargs['negative_prompt'],
			clip = clip,
		)
		
		pos, neg = pos_node.out (0), neg_node.out (0)
		
		# 4. Apply optional DWPose ControlNet conditioning
		pos, neg = self.attach_dwpose (pos, neg, vae)
		
		return model, clip, vae, pos, neg
		
	def attach_loras (self, model, clip):
		
		"""
		Sequentially chains any number of LoRA loaders onto model and clip.
		"""
		
		for lora in self.data['loras']:
			
			self.image_metadata['loras'] = []
			
			file = self.generate_local_path (lora['source'], subfolder = 'loras')
			
			weight = float (lora.get ("weight", 1.0))
			
			node = self.g.add (
				"LoraLoader",  # Or DynamicLoraLoaderFromURL
				model = model,
				clip = clip,
				lora_name = os.path.basename (file),
				strength_model = weight,
				strength_clip = float (lora.get ("clip_weight", weight))
			)
			
			new_lora = {}
			
			for key, value in lora.items ():
				
				if key == 'source':
					new_lora['name'] = Path (file).stem
				else:
					new_lora[key] = value
			
			self.image_metadata['loras'].append (new_lora)
			
			model, clip = node.out (0), node.out (1)
			
		return model, clip
	
	def attach_ip_adapter (self, model):
		
		"""
		Optionally attaches IP-Adapter conditioning if reference image is provided.
		"""
		
		if "image" in self.data["ip_adapter"]:
			
			# 1. Decode reference image from base64 in memory
			ip_ref = self.g.add ("LoadImageFromBase64", base64_str = self.data["ip_adapter"]["image"])
			
			# 2. Load unified IP-Adapter and CLIP Vision models via preset
			# Supported presets: "LIGHT - SD1.5 only (low strength)", "STANDARD (medium strength)",
			# "PLUS (high strength)", "PLUS FACE (portraits)", "FULL FACE - SD1.5 only (portraits stronger)"
			ip_loader = self.g.add (
				"IPAdapterUnifiedLoader",
				model = model,
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
			
			return ip_apply.out (0)
			
		else:
			return model
	
	def attach_dwpose (self, pos, neg, vae):
		
		"""
		Optionally extracts DWPose skeleton and applies OpenPose ControlNet.
		"""
		
		# Ветка DWPose ControlNet (активируется, если в JSON передана картинка позы)
		if "dwpose" in self.data:
			
			# 1. Загружаем референс позы из base64
			pose_ref = self.g.add (
				"LoadImageFromBase64",
				base64_str = self.data['dwpose'].get ("image")
			)
			
			detector = self.generate_local_path (
				self.data['dwpose'].get ("bbox_detector", "yolox_l.onnx"),
				subfolder = 'bbox_detectors'
			)
			
			self.image_metadata['bbox_detector'] = Path (detector).stem
			
			estimator = self.generate_local_path (
				self.data['dwpose'].get ("pose_estimator", "dw-ll_ucoco_384_bs5.onnx"),
				subfolder = 'pose_estimators'
			)
			
			self.image_metadata['pose_estimator'] = Path (estimator).stem
			
			# 2. Извлекаем скелет через DWPose
			dwpose = self.g.add (
				"DWPreprocessor",
				detect_hand = "enable",
				detect_body = "enable",
				detect_face = "enable",
				resolution = self.data["width"],
				bbox_detector = os.path.basename (detector),
				pose_estimator = os.path.basename (estimator),
				scale_stick_for_xinsr_cn = self.data['dwpose'].get ("scale_stick", "disable"),
				image = pose_ref.out (0)
			)
			
			controlnet = self.generate_local_path (
				self.data['dwpose'].get ("model", "controlnet-openpose-sdxl.safetensors"),
				subfolder = 'controlnets'
			)
			
			self.image_metadata['controlnet'] = Path (controlnet).stem
			
			# 3. Загружаем модель ControlNet OpenPose для SDXL / Illustrious
			cn_loader = self.g.add (
				"ControlNetLoader",
				control_net_name = os.path.basename (controlnet)
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
				vae = vae
			)
			
			# Перенаправляем потоки positive и negative на выход ControlNet
			return cn_apply.out (0), cn_apply.out (1)
			
		else:
			return pos, neg
		
	def attach_face_detailer (self, image, model, clip, vae, pos, neg) -> str:
		
		"""
		Optionally refines detected faces using Impact Pack FaceDetailer.
		"""
		
		if self.data["face_detailer"]:
			
			file = self.generate_local_path (
				self.data["face_detailer"].get ("model", "bbox/face_yolov8m.pt"),
				subfolder = 'face_detailers'
			)
			
			self.image_metadata['face_detailer'] = Path (file).stem
			
			# Загружаем детектор лиц (стандартный face_yolov8m.pt из Impact Pack)
			bbox_det = self.g.add (
				"UltralyticsDetectorProvider",
				model_name = os.path.basename (file),
			)
			
			detailer = self.g.add (
				"FaceDetailer",
				image = image,
				model = model,
				clip = clip,
				vae = vae,
				positive = pos,
				negative = neg,
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
			
		return image
		
	def load_ksampler (self, model, clip, vae, out, denoise, pos, neg):
		
		# 6. Run KSampler denoising
		sampler = self.g.add (
			"KSampler",
			seed = int (self.data["seed"]),
			steps = int (self.data["steps"]),
			cfg = float (self.data["cfg_scale"]),
			sampler_name = self.data["sampler"],
			scheduler = self.data["scheduler"],
			denoise = denoise,
			model = model,
			positive = pos,
			negative = neg,
			latent_image = out
		)
		
		# 7. Decode latent to pixel image and run optional FaceDetailer
		decode = self.g.add (
			"VAEDecode",
			samples = sampler.out (0),
			vae = vae
		)
		
		image = self.attach_face_detailer (decode.out (0), model, clip, vae, pos, neg)
		
		# 8. Capture output image in memory
		out = self.g.add ("ReturnImageList", images = image)
		
		self.reset_and_run (out.id)