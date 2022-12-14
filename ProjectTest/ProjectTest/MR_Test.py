def map(k,v):
	out = []
	for w in v.split(" "):
		out.append((w,1))
	return out

def reduce(k, lv):
	sum = 0
	for one in lv:
		sum += one
	return [(k,sum)]