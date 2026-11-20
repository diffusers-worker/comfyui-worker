from PIL.Image import Image

from comfy_worflow import tensor_to_pil

class ReturnMaskTuple:
	
	"""
	Strictly typed output node for Image2Mask returning a 3-element tuple.
	"""
	
	_result: tuple[Image, Image, Image] | None = None
	
	@classmethod
	def INPUT_TYPES (cls):
		
		return {
			"required": {
				"mask": ("IMAGE",),
				"inverted_mask": ("IMAGE",),
				"control_image": ("IMAGE",),
			},
		}
	
	RETURN_TYPES = ()
	FUNCTION = "capture"
	OUTPUT_NODE = True
	CATEGORY = "api"
	
	captured: bool = False
	
	def capture (self, mask, inverted_mask, control_image) -> dict:
		
		ReturnMaskTuple._result = (
			tensor_to_pil (mask[0]),
			tensor_to_pil (inverted_mask[0]),
			tensor_to_pil (control_image[0]),
		)
		
		self.captured = True
		
		return { "ui": {} }
	
	@classmethod
	def consume (cls) -> tuple[Image, Image, Image]:
		
		if cls._result is None:
			raise RuntimeError ("ReturnMaskTuple did not capture any images during execution.")
		
		out = cls._result
		
		cls._result = None
		
		return out