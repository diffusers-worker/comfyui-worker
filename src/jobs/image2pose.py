from datetime import datetime

from core import Core
from runpod_job import RunpodJob
from workflows import image2pose

class Image2Pose (RunpodJob):
	
	workflow: image2pose.Image2Pose
	
	def __init__ (self, core: Core):
		
		super ().__init__ (core)
		
		self.workflow = image2pose.Image2Pose (self.core)
		
	def process (self, tasks: list) -> list:
		
		result = []
		
		for task in tasks:
			
			control_image, mask_image, inverted_mask = self.workflow.get_pose (task)
			
			images = {
				
				'control_image': self.workflow.save_image (control_image),
				'mask_image': self.workflow.save_image (mask_image),
				'inverted_mask': self.workflow.save_image (inverted_mask),
				
			}
			
			result.append ({
				
				'images': images,
				'date': datetime.now ().isoformat (),
				
			})
		
		return result