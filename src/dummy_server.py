import asyncio

from comfy_ui.comfy_execution.server_protocol import ExecutionServer

class DummyServer (ExecutionServer):
	
	def __init__ (self):
		
		self.client_id = None
		self.last_node_id = None
		self.last_prompt_id = None
		self.loop = asyncio.new_event_loop ()
	
	def send_sync (self, event, data, sid = None):
		
		# 1. Событие перехода к новой ноде (загрузка чекпоинта, LoRA, семплер, VAE)
		if event == "executing":
			node_id = data.get ("node")
			if node_id is None:
				print ("\n[ComfyUI] Генерация графа завершена!")
			else:
				print (f"\n[ComfyUI] Выполняется нода #{node_id}...")
		
		# 2. Событие прогресса по шагам внутри KSampler
		elif event == "progress":
			
			step = data.get ("value", 0)
			total = data.get ("max", 0)
			
			if total > 0:
				
				percent = int ((step / total) * 100)
				bar = "█" * (percent // 5) + "░" * (20 - (percent // 5))
				
				print (f"\r[KSampler] |{bar}| Шаг {step}/{total} ({percent}%)", end = "", flush = True)
	
	def queue_updated (self):
		pass