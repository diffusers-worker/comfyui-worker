from datetime import datetime

from core import Core
from runpod_job import RunpodJob
from workflows import image2caption

class Image2Caption (RunpodJob):
	
	def __init__ (self, core: Core):
		super ().__init__ (core)
	
	def process (self, tasks: list) -> list:
		
		workflow = image2caption.Image2Caption (self.core)
		
		result = []
		
		for task in tasks:
			
			result.append ({
				
				'caption': workflow.get_caption (task),
				'date': datetime.now ().isoformat (),
				
			})
			
		return result