def transpose(inp, out):
	with open(inp) as f:
		lis = [x.split(",") for x in f]
	
	for x in zip(*lis):
		with open(out, "a+") as op:
			op.write( ",".join( [y for y in x] ) + "\n" )
