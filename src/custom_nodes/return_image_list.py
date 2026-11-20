from PIL.Image import Image

from comfy_worflow import tensor_to_pil

class ReturnImageList:
	
	"""
	Strictly typed output node for workflows returning a batch of images.
	"""
	
	_results: list[Image] = []
	
	@classmethod
	def INPUT_TYPES (cls):
		return { "required": { "images": ("IMAGE",) } }
	
	RETURN_TYPES = ()
	FUNCTION = "capture"
	OUTPUT_NODE = True
	CATEGORY = "api"
	
	count: int = 0
	
	def capture (self, images) -> dict:
		
		ReturnImageList._results = [tensor_to_pil (t) for t in images]
		
		self.count = len (ReturnImageList._results)
		
		return { "ui": {} }
	
	@classmethod
	def consume (cls) -> list[Image]:
		
		out = list (cls._results)
		
		cls._results.clear ()
		
		return out