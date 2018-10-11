import gzip
import cPickle as pkl


def get_dataset(file_name):
	fp=gzip.open(file_name,'rb')
	x,y=pkl.load(fp)
	fp.close()
	return x,y


file_name="../mosi_data/valid_matrix.pkl"

v_x,v_y = get_dataset(file_name)

print(v_x)
print(v_y)