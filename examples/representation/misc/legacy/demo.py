
print(dir())

a  = 10
print(dir(a))

class DEMO:
	def __init__(self):
		self.val = 10
		self._val = 10
		self.__val = 10

a = DEMO()
print(dir(a))
print(a.__val)
print(a._DEMO__val)







from odet.utils.tool import load

in_file = 'examples/data/pc_192.168.10.5_AGMT.pcap-subflow_interval=None_q_flow_duration=0.9-all.dat'
data = load(in_file)
# print(data)
