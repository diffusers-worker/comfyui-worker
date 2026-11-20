import torch

from core import Core
from workflows.image2caption import Image2Caption

core = Core ()

core.init (
	device = 'cpu',
	verbose = True,
)

workflow = Image2Caption (core)

with torch.autocast (core.config['device']), torch.inference_mode ():
	
	caption = workflow.get_caption ({
		
		'prompt': 'Describe the clothing and appearance of the person in detail',
		'vm': {'model': 'Florence-2-large2'},
		'image': core.s3_client.upload_file ('D:\\Projects\\Children Diffusion\\dataset\\images\\2623881650\\7940147637.jpg', 'images'),
		
	})
	
	core.verbose (caption)