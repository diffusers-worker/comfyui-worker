from PIL.Image import Image

import nodes as root_nodes

from custom_nodes.return_image_list import ReturnImageList
from workflows.text2image import Text2Image

class Image2Image (Text2Image):
	
	"""
	Facade workflow for image-to-image generation with optional LoRA, IP-Adapter, DWPose, and FaceDetailer.
	"""
	
	def get_images (self, data: dict) -> list[Image]:
		
		self.data = {
			
			'image': '',
			'upscale_method': 'lanczos',
			'crop': 'center',
			'denoise': 0.75,
			
		}
		
		self.init (data)
		
		if not self.data['image']:
			raise ValueError ('Image not set')
		
		# 1. Load checkpoint, optional VAE, and apply CLIP skip
		model, clip, vae, pos, neg = self.load_base_checkpoint ()
		
		# 5. Decode source image from base64 in memory
		src_img = self.g.add (
			"LoadImageFromBase64",
			base64_str = self.data["image"],
		)
		
		image_in = src_img.out (0)
		
		# Optional resize to target dimensions (ensures VAE-compatible resolution)
		if self.data["width"] > 0 and self.data["height"] > 0:
			
			scale_node = self.g.add (
				"ImageScale",
				image = image_in,
				upscale_method = self.data["upscale_method"],
				width = int (self.data["width"]),
				height = int (self.data["height"]),
				crop = self.data["crop"],
			)
			
			image_in = scale_node.out (0)
			
		# 6. Encode image into latent space and optionally repeat for batch generation
		latent_node = self.g.add (
			"VAEEncode",
			pixels = image_in,
			vae = vae,
		)
		
		latent_out = latent_node.out (0)
		
		if self.data['images_per_prompt'] > 1:
			
			batch_node = self.g.add (
				"RepeatLatentBatch",
				samples = latent_out,
				amount = self.data['images_per_prompt'],
			)
			
			latent_out = batch_node.out (0)
		
		self.load_ksampler (model, clip, vae, latent_out, self.data['denoise'], pos, neg)
		
		return ReturnImageList.consume ()
	
	def set_image_metadata (self):
		
		self.image_metadata = {
			
			'workflow': 'image2image',
			'prompt': self.image_kwargs['prompt'],
			'negative_prompt': self.image_kwargs['negative_prompt'],
			'sampler': self.data['sampler'],
			'cfg_scale': self.data['cfg_scale'],
			'cfg_rescale': self.data['cfg_rescale'],
			'clip_skip': self.data['clip_skip'],
			'steps': self.data['steps'],
			'seed': self.data['seed'],
			
		}