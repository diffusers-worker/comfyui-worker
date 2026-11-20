import traceback

import runpod

import os
import sys

sys.path.insert (0, os.path.abspath ("src/comfy_ui"))

if "--cpu" not in sys.argv:
  sys.argv.append("--cpu")

import comfy.cli_args
comfy.cli_args.args.cpu = True

# 1. Режим минимального потребления памяти (выгружает CLIP после кодирования промпта,
#    а UNet — перед декодированием VAE, аналог enable_model_cpu_offload)
comfy.cli_args.args.lowvram = True

# 2. Заставляем UNet и текстовые энкодеры работать в bfloat16/float16 вместо тяжёлого float32
#    (экономит ~50% оперативной памяти!)
comfy.cli_args.args.bf16_unet = True
comfy.cli_args.args.bf16_text_enc = True

sys.path.insert (0, '/app/site-packages')
sys.path.insert (0, '/app/src')

from core import Core

from jobs.image2caption import Image2Caption
from jobs.image2image import Image2Image
# from jobs.image2pose import Image2Pose
from jobs.text2image import Text2Image

def handler (data) -> dict:
  
  core = Core ()
  
  data = data['input']
  
  config = data['config']
  
  core.init (
    
    device = config['device'],
    verbose = config['verbose'],
    results = config.get ('results', core.config['results']),
    
    downloader = {
      
      'workspace_dir': config['downloader']['workspace_dir'],
      
      's3': {
        
        'endpoint': 'https://ca79ba62986500cb141e1fd11ca143fc.r2.cloudflarestorage.com',
        'access_key_id': '6002c3d1b14724871bfe6ec40cc779bd',
        'secret_access_key': 'cbf9d5efdfcfc9ce43d78b56c3f774f3271c9f7c7c698249dfa16bf734462abe',
        
      },
      
      'huggingface': { 'token': os.environ['HF_TOKEN'], },
      
    }
    
  )
  
  jobs = {
    
    'text2image': Text2Image,
    'image2image': Image2Image,
    # 'image2pose': Image2Pose,
    'image2caption': Image2Caption,
    
  }
  
  workflows = data['workflows']
  
  if workflows:
    
    for workflow, tasks in workflows.items ():
      
      if workflow in jobs:
        
        if tasks:
          
          job = jobs[workflow] (core)
          
          return { 'status': 'success', 'results': {workflow: job.process (tasks)}, }
          
        else:
          return { 'status': 'error', 'message': f'Tasks not found for workflow {workflow}', }
        
      else:
        return { 'status': 'error', 'message': f'Workflow {workflow} not found', }
      
  return { 'status': 'error', 'message': 'Workflows not found', }
  
if __name__ == '__main__':
  runpod.serverless.start ({ 'handler': handler })