from datetime import datetime

from core import Core
from runpod_job import RunpodJob
from workflows import image2mask

class Image2Mask (RunpodJob):
	
	workflow: image2mask.Image2Mask
	
	def __init__ (self, core: Core):
		
		super ().__init__ (core)
		
		self.workflow = image2mask.Image2Mask (self.core)
	
	def process (self, tasks: list) -> list:
		
		result = []
		
		for task in tasks:
			
			mask_image, inverted_mask, control_image = self.workflow.get_masks (task)
			
			images = {
				
				'mask_image': self.workflow.save_image (mask_image),
				'inverted_mask': self.workflow.save_image (inverted_mask),
				'control_image': self.workflow.save_image (control_image),
				
			}
			
			result.append ({
				
				'images': images,
				'date': datetime.now ().isoformat (),
				
			})
			
		return result