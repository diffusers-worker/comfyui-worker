import argparse
import json
import os

import torch

from core import Core
from modules.image import save_image
from workflows.text2image import Text2Image

core = Core ()

parser = argparse.ArgumentParser ()

parser.add_argument (
	'--image', '-i',
	action = 'append',
	help = 'Image file for Image2Image. Can be multiple.',
)

parser.add_argument (
	'--folder', '-f',
	action = 'append',
	help = 'Images folder for Image2Image. Can be multiple.',
)

parser.add_argument (
	'--device', '-d',
	type = str,
	help = f'Device ("{core.config["device"]}" by default)',
)

parser.add_argument (
	'--output-path',
	type = str,
	default = os.getcwd (),
	help = 'Images output path (current dir by default)',
)

parser.add_argument (
	'--wildcards-path',
	type = str,
	default = os.getcwd (),
	help = 'Path to wildcards folder',
)

parser.add_argument (
	'--debug',
	type = int,
	default = core.config['debug'],
	help = 'Debug mode'
)

parser.add_argument (
	'--verbose', '-v',
	action = 'store_false',
	default = core.config['verbose'],
	help = 'Verbose mode'
)

args = vars (parser.parse_args (
	# args = None if sys.argv[1:] or core.data else ['--help']
))

core.init (
	device = args['device'],
	wildcards_path = args['wildcards_path'],
	verbose = args['verbose'],
)

task = Text2Image (core)

while True:
	
	data = input ('\nEnter JSON (or \'exit\' for exit): ')
	
	if data.lower () == 'exit':
		break
	
	if not data.strip ():
		continue
	
	with torch.autocast (task.core.config['device']), torch.inference_mode ():
		
		pipe = task.get_pipeline (json.loads (data))
		
		i = 0
		
		for image in pipe.images:
			
			i += 1
			
			file = save_image (
				image,
				args['output_path'],
				task.core.file_date_name () + '_' + str (i) + '.jpg'
			)
			
			task.core.verbose (f'Image {file} saved successfully')