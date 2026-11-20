import os
from pathlib import Path

from PIL.Image import Image

import nodes as root_nodes
from comfy_ui import nodes as pkg_nodes
from comfy_worflow import ComfyWorkflow
from custom_nodes.return_mask_tuple import ReturnMaskTuple

class Image2Pose (ComfyWorkflow):
	
	def get_pose (self, data: dict) -> tuple[Image, Image, Image]:
		
		"""
		Standalone facade for DWPose extraction without diffusion overhead.
		"""
		
		self.data = {
			
			'pose_detector': {},
			
		}
		
		self.init (data)
		
		for mod in (root_nodes, pkg_nodes):
			
			mod.NODE_CLASS_MAPPINGS.update ({
				"ReturnMaskTuple": ReturnMaskTuple,
			})
		
		# 1. Decode input image from base64
		img = self.g.add ("LoadImageFromBase64", base64_str = self.data["image"])
		
		detector = self.generate_local_path (
			self.data['dwpose'].get ("bbox_detector", "yolox_l.onnx"),
			subfolder = 'bbox_detectors'
		)
		
		self.image_metadata['bbox_detector'] = Path (detector).stem
		
		estimator = self.generate_local_path (
			self.data['dwpose'].get ("pose_estimator", "dw-ll_ucoco_384_bs5.onnx"),
			subfolder = 'pose_estimators'
		)
		
		self.image_metadata['pose_estimator'] = Path (estimator).stem
		
		# 2. Run DWPose preprocessor
		# Outputs: slot 0 -> IMAGE (skeleton), slot 1 -> MASK (pose/body silhouette mask)
		dwpose = self.g.add (
			"DWPreprocessor",
			detect_hand = self.data['pose_detector'].get ("detect_hand", "enable"),
			detect_body = self.data['pose_detector'].get ("detect_body", "enable"),
			detect_face = self.data['pose_detector'].get ("detect_face", "enable"),
			resolution = self.data["width"],
			bbox_detector = os.path.basename (detector),
			pose_estimator = os.path.basename (estimator),
			scale_stick_for_xinsr_cn = "disable",
			image = img.out (0)
		)
		
		# 3. Convert raw MASK (slot 1) to RGB IMAGE for mask_image
		mask_img = self.g.add ("MaskToImage", mask = dwpose.out (1))
		
		# 4. Invert raw MASK and convert to RGB IMAGE for inverted_mask
		inv_mask = self.g.add ("InvertMask", mask = dwpose.out (1))
		inv_mask_img = self.g.add ("MaskToImage", mask = inv_mask.out (0))
		
		# 5. Capture all three outputs in memory under their respective keys
		out = self.g.add (
			"ReturnMaskTuple",
			control_image = dwpose.out (0),
			mask_image = mask_img.out (0),
			inverted_mask = inv_mask_img.out (0),
		)
		
		self.reset_and_run (out.id)
		
		return ReturnMaskTuple.consume ()
		
	def set_image_metadata (self):
		
		self.image_metadata = {
			
			'workflow': 'image2pose',
			
		}