import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from core import Core
from modules.image import load_image
from workflow import Workflow

class Image2Caption (Workflow):
	
	processor = None
	
	current_vm: dict = {}
	
	def __init__ (self, core: Core):
		super ().__init__ (core)
	
	def get_caption (self, data: dict) -> str:
		
		self.data.update ({
			
			'vm': {},
			
		})
		
		self.init (data)
		
		if self.data['vm'] != self.current_vm:
			
			self.core.verbose ('Loading vision model...')
			
			model_kwargs = self.model_kwargs ()
			
			model_kwargs['backend'] = 'torchvision'
			
			source = self.generate_local_path (self.data['vm']['model'], subfolder = 'vm')
			
			self.processor = AutoProcessor.from_pretrained (
				source,
				**model_kwargs
			)
			
			self.pipe = AutoModelForCausalLM.from_pretrained (
				source,
				**self.model_kwargs ()
			).to (self.core.config['device'])
			
			self.pipe.eval ()
			
			self.current_vm = self.data['vm']
			
		self.core.verbose ('Generating caption...')
		
		input_image = load_image (self.data['image'])
		
		inputs = self.processor (
			text = self.process_prompt (self.data['prompt']),
			images = input_image,
			return_tensors = 'pt'
		).to (self.core.config['device'])
		
		with torch.no_grad ():
			
			generated_ids = self.pipe.generate (
				input_ids = inputs['input_ids'],
				pixel_values = inputs['pixel_values'],
				max_new_tokens = 1024,
				num_beams = 3,  # Low beam search keeps it fast yet accurate on CPU
				do_sample = False
			)
			
		generated_text = self.processor.batch_decode (
			generated_ids,
			skip_special_tokens = True
		)[0]
		
		return generated_text.strip ()