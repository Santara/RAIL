import numpy as np 
import h5py
import matplotlib.pyplot as plt 
import argparse
import pandas as pd 
from scipy import signal 
import seaborn as sns
from IPython.core.debugger import Pdb

EXPERT_MEAN = 9790.99989735
EXPERT_STD = 1175.06649136

def moving_average(inp,window_size):
	filt = np.ones((window_size))
	filt = filt/len(filt)
	out = np.convolve(inp, filt, "same")
	return out

def plot_log_file(filename, fields_to_plot, save_dir, use_moving_average, moving_average_window_size):
	f = h5py.File(filename,'r')
	log = f['log']
	field_names = [ "iter", "trueret", "iret", "trueret_std", "ire_std", "dl", "pgrad", "rloss", "racc", "rgrad"]
	plt.figure()
	handles=[]
	for field in fields_to_plot:
		data = [log[i][field] for i in range(log.shape[0])]
		if use_moving_average:
			data = moving_average(data, window_size=moving_average_window_size)
		tmp, = plt.plot(data, label=field_names[field])
		handles.append(tmp)
	tmp, = plt.plot(EXPERT_STD*np.ones((1500,)), label="expert")
	handles.append(tmp)
	plt.legend(handles=handles)
	plt.grid(True)
	plt.savefig(save_dir+filename.split('/')[-1][:-3]+'.png')
	plt.show()

def plot_log_file_fancy(filename, save_dir, use_moving_average, use_median_filtering, moving_average_window_size):
	f = h5py.File(filename,'r')
	log = f['log']
	field_names = [ "iter", "trueret", "iret", "trueret_std", "ire_std", "dl", "pgrad", "rloss", "racc", "rgrad"]
	fields_to_plot = [field_names[i] for i in [1,3]]
	
	mean = [log[i][1] for i in range(log.shape[0])]
	if use_moving_average:
		mean = moving_average(mean, window_size=moving_average_window_size)
	elif use_median_filtering:
		mean = signal.medfilt(mean, window_size)
	
	std = [log[i][3] for i in range(log.shape[0])]
	if use_moving_average:
		std = moving_average(std, window_size=moving_average_window_size)
	elif use_median_filtering:
		std = signal.medfilt(std, window_size)
	
	plt.figure()
	plt.plot(mean)
	plt.fill_between(xrange(mean.shape[0]), [i+j for (i,j) in zip(mean,std)], [i-j for (i,j) in zip(mean,std)], facecolor='b', alpha=0.25)
	plt.grid(True)
	plt.savefig(save_dir+filename.split('/')[-1][:-3]+'_fancy.png')
	plt.show()

def plot_log_file_mean_with_raw(filename, save_dir, use_moving_average, use_median_filtering, moving_average_window_size):
	f = h5py.File(filename,'r')
	log = f['log']
	field_names = [ "iter", "trueret", "iret", "trueret_std", "ire_std", "dl", "pgrad", "rloss", "racc", "rgrad"]
	fields_to_plot = [field_names[i] for i in [1,3]]
	
	mean_raw = [log[i][1] for i in range(log.shape[0])]
	if use_moving_average:
		mean = moving_average(mean_raw, window_size=moving_average_window_size)
	elif use_median_filtering:
		mean = signal.medfilt(mean_raw, window_size)
	
	std_raw = [log[i][3] for i in range(log.shape[0])]
	if use_moving_average:
		std = moving_average(std_raw, window_size=moving_average_window_size)
	elif use_median_filtering:
		std = signal.medfilt(std_raw, window_size)
	
	plt.figure()
	plt.plot(mean, color='g')
	plt.plot(mean_raw, color='g', alpha=0.3)
	plt.plot(std, color='b')
	plt.plot(std_raw, color='b', alpha=0.3)
	plt.grid(True)
	plt.savefig(save_dir+filename.split('/')[-1][:-3]+'_withRaw.png')
	# plt.show()


def compare_methods(filename1, filename2, fields_to_plot, save_dir, use_moving_average, use_median_filtering, window_size):
	f1 = h5py.File(filename1,'r')
	f2 = h5py.File(filename2,'r')
	log1 = f1['log']
	log2 = f2['log']
	field_names = [ "iter", "trueret", "iret", "trueret_std", "ire_std", "dl", "pgrad", "rloss", "racc", "rgrad"]
	for field in fields_to_plot:
		plt.figure()
		handles=[]
		data1 = [log1[i][field] for i in range(log1.shape[0])]
		if use_moving_average:
			data1 = moving_average(data1, window_size=window_size)
		elif use_median_filtering:
			data1 = signal.medfilt(data1, window_size)
		tmp, = plt.plot(data1, label=filename1.split('/')[-1])
		handles.append(tmp)
		data2 = [log2[i][field] for i in range(log2.shape[0])]
		if use_moving_average:
			data2 = moving_average(data2, window_size=window_size)
		elif use_median_filtering:
			data2 = signal.medfilt(data2, window_size)
		tmp, = plt.plot(data2, label=filename2.split('/')[-1])
		handles.append(tmp)
		plt.legend(handles=handles)
		plt.grid(True)
		plt.savefig(save_dir+filename1.split('/')[-1][:-3]+'--vs--'+filename2.split('/')[-1][:-3]+'-'+field_names[field]+'.png')
		plt.show()

def write_csv(in_filename=None, out_filename=None):
	f_in = h5py.File(in_filename,'r')
	log = f_in['log']
	f_out = open(out_filename,'w')
	# field_names = [ "iter", "trueret", "iret", "trueret_std", "ire_std", "nu", "Lambda", "dl", "pgrad", "rloss", "racc", "rgrad"]
	field_names = ["iter", "trueret", "iret", "trueret_std", "ire_std", "dl", "rloss", "racc", "rgrad"]
	# print fields
	for field in field_names:
		f_out.write(field+',')
	f_out.write('\n')

	# print the log data line by line
	for line_num in range(log.shape[0]):
		for entry in log[line_num]:
			f_out.write(str(entry)+',')
		f_out.write('\n')
	f_out.close()


def plot_csv(file_name, use_moving_average, moving_average_window_size, plot_all_fields, plot_fields, save_dir, expert_level):
	df = pd.read_csv(file_name, sep=',')
	titles = list(df)
	plt.figure()
	handles = []
	if plot_all_fields:
		for i in range(len(titles)):
			data2plot = df.iloc[:,i]
			if use_moving_average:
				data2plot = moving_average(data2plot, window_size=moving_average_window_size)[:-moving_average_window_size]
			tmp, = plt.plot(data2plot, label=titles[i])
			handles.append(tmp)
	else:
		for i in plot_fields:
			data2plot = df.iloc[:,i]
			if use_moving_average:
				data2plot = moving_average(data2plot, window_size=moving_average_window_size)[:-moving_average_window_size]
			tmp, = plt.plot(data2plot, label=titles[i])
			handles.append(tmp)
	tmp, = plt.plot(expert_level*np.ones((1500,)), label="expert")
	handles.append(tmp)
	plt.legend(handles=handles)
	plt.grid(True)
	plt.savefig(save_dir+file_name.split('/')[-1][:-4]+'.png')
	plt.show()


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--plot_single', action='store_true')
	parser.add_argument('--plot_fancy', action='store_true')
	parser.add_argument('--plot_fancy_raw', action='store_true')
	parser.add_argument('--file_name', type=str, help=".h5 file path if --plot_single or --write_csv and .csv file path if --plot_csv")
	parser.add_argument('--compare', action='store_true')
	parser.add_argument('--file_name1', type=str, help=".h5 file path 1 if --compare")
	parser.add_argument('--file_name2', type=str, help=".h5 file path 2 if --compare")
	parser.add_argument('--save_dir', type=str, default="./", help="directory for saving plots")
	parser.add_argument('--use_moving_average', action='store_true')
	parser.add_argument('--use_median_filtering', action='store_true')
	parser.add_argument('--moving_average_window_size', type=int, default=3)
	parser.add_argument('--write_csv', action='store_true')
	parser.add_argument('--plot_csv', action='store_true')
	parser.add_argument('--plot_all_fields', action='store_true')
	#TODO: fix the following line - add separate conditions for different fields
	parser.add_argument('--expert_level', type=float, default=EXPERT_STD)

	args = parser.parse_args()
	fields_to_plot = [1,3]
	if args.plot_single:
		plot_log_file(args.file_name, fields_to_plot=fields_to_plot, save_dir=args.save_dir, use_moving_average=args.use_moving_average, moving_average_window_size=args.moving_average_window_size)
	elif args.compare:
		compare_methods(filename1=args.file_name1, filename2=args.file_name2, fields_to_plot=fields_to_plot, save_dir=args.save_dir, use_moving_average=args.use_moving_average, use_median_filtering=args.use_median_filtering, window_size=args.moving_average_window_size)
	elif args.write_csv:
		out_filename = args.save_dir+args.file_name.split('/')[-1][:-3]+'.csv'
		write_csv(args.file_name, out_filename)
	elif args.plot_csv:
		plot_fields = [0,1,-1]
		plot_csv(file_name=args.file_name, use_moving_average=args.use_moving_average, moving_average_window_size=args.moving_average_window_size, plot_all_fields=args.plot_all_fields, plot_fields=plot_fields, save_dir=args.save_dir, expert_level=args.expert_level)
	elif args.plot_fancy:
		plot_log_file_fancy(args.file_name, save_dir=args.save_dir, use_moving_average=args.use_moving_average, use_median_filtering=args.use_median_filtering, moving_average_window_size=args.moving_average_window_size)
	elif args.plot_fancy_raw:
		plot_log_file_mean_with_raw(args.file_name, save_dir=args.save_dir, use_moving_average=args.use_moving_average, use_median_filtering=args.use_median_filtering, moving_average_window_size=args.moving_average_window_size)
