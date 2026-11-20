import torch

from core import Core
from modules.image import save_image
from workflows.image2pose import Image2Pose

core = Core ()

core.init (
	device = 'cpu',
	verbose = True
)

workflow = Image2Pose (core)

with torch.autocast (core.config['device']), torch.inference_mode ():
	
	image = workflow.get_pose ({
		'image': 'D:\\Projects\\Children Diffusion\\dataset\\images\\2623881650\\7940147637.jpg',
	})
	
	file = save_image (
		image,
		'D:\\Projects\\Children Diffusion\\dataset\\images\\2623881650',
		workflow.core.file_date_name () + '.jpg'
	)
	
	core.verbose (f'Pose image {file} saved successfully')