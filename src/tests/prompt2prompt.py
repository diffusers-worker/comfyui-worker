from workflows.prompt2prompt import Prompt2Prompt

Prompt2Prompt (
	base_model = 'SDXLTurbo',
	model = 'D:\\Models\\sdxl-turbo-openvino-int8',
	prompt = '''Adorable infant playing with a variety of {colorful|big} rattle toys.''',
).process ()