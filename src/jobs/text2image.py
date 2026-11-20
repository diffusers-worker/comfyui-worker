from datetime import datetime

from core import Core
from runpod_job import RunpodJob
from workflows import text2image

class Text2Image (RunpodJob):
	
	workflow: text2image.Text2Image
	
	def __init__ (self, core: Core):
		
		super ().__init__ (core)
		
		self.workflow = text2image.Text2Image (self.core)
		
	def process (self, tasks: list) -> list:
		
		result = []
		
		for task in tasks:
			
			images = []
			
			for idx, image in enumerate (self.workflow.get_images (task)):
				
				images.append ({
					
					'image': self.workflow.save_image (image),
					
				})
			
			result.append ({
				
				'images': images,
				'date': datetime.now ().isoformat (),
				
			})
			
		return result