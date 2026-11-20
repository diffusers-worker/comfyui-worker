import os

from optimum.intel import OVStableDiffusionXLPipeline
import torch

from core import Core
from modules.image import save_image
from workflows.text2image import Text2Image

core = Core ()

core.init (
	
	device = 'cpu',
	verbose = True,
	
	downloader = {
		
		'workspace_dir': 'F:\\Models',
		
		's3': {
			
			'endpoint': 'https://ca79ba62986500cb141e1fd11ca143fc.r2.cloudflarestorage.com',
			'access_key_id': '6002c3d1b14724871bfe6ec40cc779bd',
			'secret_access_key': 'cbf9d5efdfcfc9ce43d78b56c3f774f3271c9f7c7c698249dfa16bf734462abe',
			
		},
		
		'huggingface': { 'token': os.environ['HF_TOKEN'], }
		
	}
	
)

workflow = Text2Image (core)

with torch.autocast (workflow.core.config['device']), torch.inference_mode ():
	
	pipe = workflow.get_pipeline (
		{
			'model': 'rupeshs/SDXL-Lightning-2steps-openvino-int8',
			'scheduler': 'UniPC',
			'steps': 2,
			'seed': 42,
			'width': 512,
			'height': 512,
			'cfg_scale': 4,
			'prompt': '''Adorable infant playing with a variety of colorful rattle toys.''',
		},
		pipeline = OVStableDiffusionXLPipeline
	)
	
	i: int = 0
	
	for image in pipe.images:
		
		i += 1
		
		file = save_image (
			image,
			'D:\\Новая папка\\output',
			workflow.core.file_date_name () + '_' + str (i) + '.jpg'
		)
		
		workflow.core.verbose (f'Image {file} saved successfully')