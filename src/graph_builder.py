class NodeRef:
	
	def __init__ (self, node_id: str):
		self.id = node_id
	
	def out (self, slot: int) -> list:
		return [self.id, slot]

class GraphBuilder:
	
	nodes = {}
	seq = 1
	
	def add (self, class_type: str, **inputs) -> NodeRef:
		
		node_id = str (self.seq)
		
		self.seq += 1
		
		self.nodes[node_id] = {
			
			"class_type": class_type,
			"inputs": inputs,
			
		}
		
		return NodeRef (node_id)