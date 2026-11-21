from abc import abstractmethod

import PIL
from PIL.Image import Image
import numpy as np

from comfy_direct_runner import ComfyDirectRunner
from graph_builder import GraphBuilder
from workflow import Workflow

class ComfyWorkflow (Workflow):
	
	g: GraphBuilder
	
	runner = ComfyDirectRunner ()
	
	def reset_and_run (self, output_node_id: str) -> None:
		
		"""
		Executes the built graph via the direct runner.
		"""
		
		self.runner.run_graph (self.g.nodes, [output_node_id])
	
	@abstractmethod
	def set_image_metadata (self):
		pass
	
	def init (self, data: dict):
		
		super ().init (data)
		
		self.g = GraphBuilder ()  # We need to create new graph at every query
	
def tensor_to_pil (img_tensor) -> Image:
	
	"""
	Converts a single ComfyUI image tensor [H, W, C] to a PIL Image.
	"""
	
	arr = np.clip (255.0 * img_tensor.cpu ().numpy (), 0, 255).astype (np.uint8)
	
	return PIL.Image.fromarray (arr)