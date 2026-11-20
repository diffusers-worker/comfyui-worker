import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

import torch

from comfy_ui import execution
from comfy_ui.comfy import model_management
from comfy_ui.comfy.cli_args import args
from dummy_server import DummyServer

class ComfyDirectRunner:
	
	def __init__ (self):
		
		cache_ram = 0
		cache_ram_inactive = 0
		
		if not args.cache_classic and not args.cache_none and args.cache_lru <= 0:
			
			cache_ram = min (10.0, max (2.0, model_management.total_ram * 0.10 / 1024.0))
			cache_ram_inactive = min (128.0, model_management.total_ram / 1024.0)
			
			if len (args.cache_ram) > 0:
				cache_ram = args.cache_ram[0]
			if len (args.cache_ram) > 1:
				cache_ram_inactive = args.cache_ram[1]
		
		cache_type = execution.CacheType.RAM_PRESSURE
		
		if args.cache_classic:
			cache_type = execution.CacheType.CLASSIC
		elif args.cache_lru > 0:
			cache_type = execution.CacheType.LRU
		elif args.cache_none:
			cache_type = execution.CacheType.NONE
		
		self.executor = execution.PromptExecutor (
			server = DummyServer (),
			cache_type = cache_type,
			cache_args = { "lru": args.cache_lru, "ram": cache_ram, "ram_inactive": cache_ram_inactive }
		)
		
		self._thread_pool = ThreadPoolExecutor (max_workers = 1)
		
	@torch.inference_mode ()
	def run_graph (self, workflow: dict, output_node_ids: list[str]):
		
		"""
		Executes a compiled ComfyUI workflow synchronously via isolated event loop.
		"""
		
		future = self._thread_pool.submit (
			asyncio.run,
			self.executor.execute_async (
				workflow,
				str (uuid.uuid4 ()),
				extra_data = {},
				execute_outputs = output_node_ids
			)
		)
		
		future.result ()