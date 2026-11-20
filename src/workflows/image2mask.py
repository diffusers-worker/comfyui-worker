import os
from pathlib import Path

from PIL.Image import Image

import nodes as root_nodes
from comfy_ui import nodes as pkg_nodes
from comfy_worflow import ComfyWorkflow

from custom_nodes.return_mask_tuple import ReturnMaskTuple

class Image2Mask (ComfyWorkflow):
	
	def get_masks (self, data: dict) -> tuple[Image, Image, Image]:
		
		self.data = {
			
			'sam': {},
			
		}
		
		self.core.verbose ('Generating mask...')
		
		self.init (data)
		
		for mod in (root_nodes, pkg_nodes):
			
			mod.NODE_CLASS_MAPPINGS.update ({
				"ReturnMaskTuple": ReturnMaskTuple,
			})
		
		# 1. Decode source image from base64 in memory
		img = self.g.add ("LoadImageFromBase64", base64_str = self.data['sam']["image"])
		
		file = self.generate_local_path (
			self.data['sam'].get ("model", "sam_vit_b_01ec64.pth"),
			subfolder = 'sam_models'
		)
		
		self.image_metadata['model'] = Path (file).stem
		
		# 2. Load Segment Anything Model (SAM)
		# Common models: "sam_vit_b_01ec64.pth", "sam_vit_l_0b3195.pth", "sam_vit_h_4b8939.pth"
		sam_loader = self.g.add (
			"SAMLoader",
			model_name = os.path.basename (file),
			device_mode = self.data['sam'].get ("device_mode", "AUTO")
		)
		
		file = self.generate_local_path (
			self.data['sam'].get ("detector_model", "bbox/person_yolov8m-seg.pt"),
			subfolder = 'detector_models'
		)
		
		self.image_metadata['detector_model'] = Path (file).stem
		
		# 3. Load YOLO object detector to locate target bounding boxes
		# Common models: "bbox/face_yolov8m.pt", "bbox/person_yolov8m-seg.pt", "segm/deepfashion2_yolov8s-seg.pt"
		bbox_detector = self.g.add (
			"UltralyticsDetectorProvider",
			model_name = os.path.basename (file)
		)
		
		# 4. Run bounding box detection on the input image
		segs = self.g.add (
			"BboxDetectorSEGS",
			threshold = float (self.data['sam'].get ("bbox_threshold", 0.5)),
			dilation = int (self.data['sam'].get ("dilation", 10)),
			crop_factor = float (self.data['sam'].get ("crop_factor", 1.5)),
			drop_size = int (self.data['sam'].get ("drop_size", 10)),
			labels = self.data['sam'].get ("labels", "all"),
			bbox_detector = bbox_detector.out (0),
			image = img.out (0)
		)
		
		# 5. Refine segmentation with SAM and output combined binary mask
		sam_mask = self.g.add (
			"SAMDetectorCombined",
			detection_hint = self.data['sam'].get ("detection_hint", "center-1"),
			dilation = int (self.data['sam'].get ("mask_dilation", 0)),
			threshold = float (self.data['sam'].get ("sam_threshold", 0.93)),
			bbox_expansion = int (self.data['sam'].get ("bbox_expansion", 0)),
			mask_hint_threshold = float (self.data['sam'].get ("mask_hint_threshold", 0.7)),
			mask_hint_use_negative = self.data['sam'].get ("mask_hint_use_negative", "False"),
			sam_model = sam_loader.out (0),
			segs = segs.out (0),
			image = img.out (0)
		)
		
		mask_img = self.g.add ("MaskToImage", mask = sam_mask.out (0))
		inv_mask = self.g.add ("InvertMask", mask = sam_mask.out (0))
		inv_mask_img = self.g.add ("MaskToImage", mask = inv_mask.out (0))
		
		preview = self.g.add (
			"SEGSPreview",
			segs = segs.out (0),
			image = img.out (0),
			alpha_mode = True,
			min_alpha = 0.2
		)
		
		out = self.g.add (
			"ReturnMaskTuple",
			mask_image = mask_img.out (0),
			inverted_mask = inv_mask_img.out (0),
			control_image = preview.out (0),
		)
		
		self.reset_and_run (out.id)
		
		return ReturnMaskTuple.consume ()
	
	def set_image_metadata (self):
		
		self.image_metadata = {
			
			'workflow': 'text2mask',
			
		}