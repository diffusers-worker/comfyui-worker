from abc import ABC, abstractmethod

from core import Core

class RunpodJob (ABC):
	
	core: Core
	
	def __init__ (self, core: Core):
		self.core = core
	
	@abstractmethod
	def process (self, tasks: list) -> list:
		pass