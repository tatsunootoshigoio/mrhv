#-----------------------------------------------------#
# MR & Hall voltage vs. 2theta ploter v0.1		  ----#
# author: tatsunootoshigo, 7475un00705hi90@gmail.com  #
#-----------------------------------------------------#

# Imports
import numpy as np
import peakutils
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy import stats
from sklearn.preprocessing import MinMaxScaler as sklscl
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter
from matplotlib.path import Path

# script version
version = '0.1b'
version_name = 'mrhv_ploter_v' + version + '.py'

# some useful constants
cm = 1e-2
nm = 1e-9
mu0 = 4*np.pi*1e-7
mega = 1e+6
kilo = 1e+3
mili = 1e-3
micro = 1e-6
tesla = 1e+4 # Gauss to tesla

# =================== SAMPLE DATA STARTS HERE =================
# date of the measurement, sample id, composition label, etc.
sample_mdate = '300818'
sample_id = 'E2488'
sample_no = 0
dataset_no = 1

slayer5 = 	r'$-$'
slayer4 = 	r'$-$'
slayer3 = 	r'$-$'
slayer2 = 	r'$AlO$'
slayer1 = 	r'$Fe-Si$'
substrate = r'$Al_2O_3$'
sdelimiter = r'$\rfloor\lfloor$'

# sample dimensions
#sample_w = 10.0 # mm
#sample_h = 10.0 # mm
#sample_l = 0.5 # mm
#sample_th = 3.0 # nm

# measurement current
sample_current = 0.1 # / mA
sample_field = 2.0 # / T

# sample resistivity parameters
sample_w = 0.2 # mm
sample_l = 0.7 # mm
sample_d = 3.0  # nm
sample_V = 173.6 # mV

# sample hall electrode distance
sample_vh_l = sample_w # mm (large ones, 3x3 grid)
sample_vh_w = 0.1 # mm
#sample_vh_l = 0.8 # mm (small ones, 4x4 grid)

# hall voltage
sample_vh = 1.83 # mV

# =================== SAMPLE DATA ENDS HERE ==================

# sample resistivity in mu*Ohm*cm
sample_rho = (sample_V / sample_current) * (sample_w * sample_d) / sample_l / 10.0

# sample hall_rho in mu*Ohm*cm
sample_hall_rho = (sample_vh / sample_current) * sample_d / 10.0

# output pdf, svg, tsv file names
out_pdf = 'mrhv_plots_' + sample_id + '-' + np.str(sample_no) + '_' + np.str(dataset_no) + '.pdf'
out_svg = 'mrhv_plots_' + sample_id + '-' + np.str(sample_no) + '_' + np.str(dataset_no) + '.svg'
out_tsv = 'mrhv_plots_summary_' + sample_mdate + sample_id + '_' + np.str(dataset_no) + '.tsv'

# plot legend lables
mr_label_yz = r'$zy\;\;\left(H_z \rightarrow H_y \rightarrow H_z\rightarrow H_y \rightarrow H_z\right)$'
mr_label_xy = r'$xy\;\;\left(H_x \rightarrow H_y \rightarrow H_x\rightarrow H_y \rightarrow H_x\right)$'
mr_label_xz = r'$zx\;\;\left(H_z \rightarrow H_x \rightarrow H_z\rightarrow H_x \rightarrow H_z\right)$'

hv_label_yz = r'$zy$'
hv_label_xy = r'$xy$'
hv_label_xz = r'$zx$'

label_xy_fit = r'$xy fit$'
label_ohm = r'$\;\Omega$'

# axes labels for plots
axis_label_th = r'$thickness\, / \, nm$'
axis_label_theta = r'$\theta\, / \, \circ$'
axis_label_volt = r'$V$'
axis_label_ohm = r'$R\;/\;\Omega$'
axis_label_points = r'$point\;no.$' 

# number of decimal points to display in tick labels
xprec = 0
yprec = 1

# number of decimal places to display strings
sprec = 4

# set axes range to some nice round value
xmin = 0.0
xmax = 360.0 # 2*np.pi

# y axis volatge min max for mr curves
mr_ymin = 1733.0 # mV
mr_ymax = 1735.4

# y axis voltage min and max for hv curves
hv_ymin = -24 # mV
hv_ymax = 24

# shift the plots vertically
mr_shift_xy = -0.001
mr_shift_xz = 0.45
mr_shift_yz = 0

hv_shift_xy = 0.0
hv_shift_xz = 0.0
hv_shift_yz = 0.0

# fitting parameters for xy plot
R0 = 15.105
A1 = 0.0001
A2 = 0.0048
d1 = -0.05
d2 = -0.05

# position of graph description
desc_x = 0.17
desc_y = 0.04

def set_plot_title(plot_no):
	if plot_no == 1:
		# sample label describing the layers
		sample_label = '[' + substrate + sdelimiter + slayer1 + sdelimiter + slayer2 + ']'
		# plot title using sample label
		plt_title = 'id: ' + sample_id + '-' + np.str(sample_no) + ' ' + sample_label
	if plot_no == 2:

		plt_title = r'$R_{AHE}$' + ' angular dependance'
	
	ax =plt.gca()
	ax.set_title(plt_title, loc='right', fontsize=14)
	
	return;
	
def custom_axis_formater(custom_x_label, custom_y_label, xmin, xmax, ymin, ymax, xprec, yprec):
	
	# get axes and tick from plot 
	ax = plt.gca()

	# set the number of major and minor ticks for x,y axes
	# prune='lower' --> remove lowest tick label from x axis
	xmajorLocator = MaxNLocator(7, prune='lower') 
	xmajorFormatter = FormatStrFormatter('%.'+ np.str(xprec) + 'f')
	xminorLocator = MaxNLocator(14) 
	
	ymajorLocator = MaxNLocator(15) 
	ymajorFormatter = FormatStrFormatter('%.'+ np.str(yprec) + 'f')
	yminorLocator = MaxNLocator(30)

	ax.xaxis.set_major_locator(xmajorLocator)
	ax.yaxis.set_major_locator(ymajorLocator)

	ax.xaxis.set_major_formatter(xmajorFormatter)
	ax.yaxis.set_major_formatter(ymajorFormatter)

	# for the minor ticks, use no labels; default NullFormatter
	ax.xaxis.set_minor_locator(xminorLocator)
	ax.yaxis.set_minor_locator(yminorLocator)
	
	# format major and minor ticks width, length, direction 
	ax.tick_params(which='both', width=1, direction='in', labelsize=14)
	ax.tick_params(which='major', length=6)
	ax.tick_params(which='minor', length=4)

	# set axes thickness
	ax.spines['top'].set_linewidth(1)
	ax.spines['bottom'].set_linewidth(1)
	ax.spines['right'].set_linewidth(1)
	ax.spines['left'].set_linewidth(1)

	# grid and axes are drawn below the data plot
	ax.set_axisbelow(True)

	# add x,y grids to plot area
	ax.xaxis.grid(True, zorder=0, color='lavender', linestyle='-', linewidth=1)
	ax.yaxis.grid(True, zorder=0, color='lavender', linestyle='-', linewidth=1)

	# set axis labels
	ax.set_xlabel(custom_x_label, fontsize=14)
	ax.set_ylabel(custom_y_label, fontsize=14)

	# set plot title
	#ax.set_title(custom_title, loc='right', fontsize=14)

	#for label in ax.get_xticklabels():
	#	label.set_fontsize(12)
	#	label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))
		#plt.xticks([-1.5*np.pi, -np.pi, -np.pi/2, 0, np.pi/2, np.pi, 1.5*np.pi, 2*np.pi],
			#[r'$-\frac{3}{2}\pi$',r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$', r'$\frac{3}{2}\pi$'])
	#	plt.xticks([0, np.pi/2, np.pi, 1.5*np.pi, 2*np.pi],
	#		[r'$0$', r'$+\pi/2$', r'$+\pi$', r'$\frac{3}{2}\pi$'])

	return;

def tuteta_gen_desc(sample_field, sample_current, cal_resist_in):
	
	row_delimiter = '\n'
	hr_line = '\n------------------------------------------\n'
	separator = r'$\;\;\;$'

	Rxy1 = np.str(np.round(cal_resist_in[0][0], sprec))
	Rxy2 = np.str(np.round(cal_resist_in[0][1], sprec))
	Rxy3 = np.str(np.round(cal_resist_in[0][2], sprec))
	Rxy4 = np.str(np.round(cal_resist_in[0][3], sprec))
	Rxy5 = np.str(np.round(cal_resist_in[0][4], sprec))
	MRxy = np.str(np.round(cal_resist_in[0][5], sprec))

	Rzx1 = np.str(np.round(cal_resist_in[1][0], sprec))
	Rzx2 = np.str(np.round(cal_resist_in[1][1], sprec))
	Rzx3 = np.str(np.round(cal_resist_in[1][2], sprec))
	Rzx4 = np.str(np.round(cal_resist_in[1][3], sprec))
	Rzx5 = np.str(np.round(cal_resist_in[1][4], sprec))
	MRzx = np.str(np.round(cal_resist_in[1][5], sprec))

	Rzy1 = np.str(np.round(cal_resist_in[2][0], sprec))
	Rzy2 = np.str(np.round(cal_resist_in[2][1], sprec))
	Rzy3 = np.str(np.round(cal_resist_in[2][2], sprec))
	Rzy4 = np.str(np.round(cal_resist_in[2][3], sprec))
	Rzy5 = np.str(np.round(cal_resist_in[2][4], sprec))
	MRzy = np.str(np.round(cal_resist_in[2][5], sprec))

	rhoxx = np.str(np.round(sample_rho, sprec))
	rhoxy = np.str(np.round(sample_hall_rho, sprec))
	theta_ahe = np.str(np.round(sample_hall_rho / sample_rho, sprec))
	theta_ahe2 = np.str(np.round((sample_hall_rho / sample_rho)**2, sprec))

	plt.figtext(desc_x, desc_y, 'R peak values' + hr_line +  r'$R_{xy}\;/\;\Omega$' + separator + separator + separator + r'$R_{zx}\;/\;\Omega$' + separator + separator + separator + r'$R_{zy}\;/\;\Omega$' + row_delimiter + Rxy1 + separator + Rzx1 + separator + Rzy1 + row_delimiter + Rxy2 + separator + Rzx2 + separator + Rzy2 + row_delimiter + Rxy3 + separator + Rzx3 + separator + Rzy3 + row_delimiter + Rxy4 + separator + Rzx4 + separator + Rzy4 + row_delimiter + Rxy5 + separator + Rzx5 + separator + Rzy5 + hr_line + r'$MR_{ratio}^{(xy)}$' + separator + separator + r'$MR_{ratio}^{(zx)}$' + separator + separator + r'$MR_{ratio}^{(zy)}$' + row_delimiter + MRxy + '%' + separator + MRzx + '%' + separator + MRzy + '%' + hr_line + r'$\rho_{xx}=$' + rhoxx + r'$\,\mu\Omega cm $' + row_delimiter + r'$\rho_{xy}=$' + rhoxy + r'$\,\mu\Omega cm $' + row_delimiter + r'$\theta_{AH}=$' + theta_ahe + separator + r'$\theta_{AH}^2=$' + theta_ahe2 + hr_line + r'$I=$'+ np.str(sample_current) + r'$\, mA$' + ', ' + r'$H=$' + np.str(sample_field) + r'$\,T$' + row_delimiter + r'$w=$' + np.str(sample_w) + r'$\;mm$'+ ', ' + r'$d=$' + np.str(sample_d) + r'$\;nm$' + ', ' + r'$L_V= $' + np.str(sample_l) + r'$\;mm$' + ', '+ r'$L_H$=' + np.str(sample_vh_l) + r'$\;mm$', size=14)
	
	# verison name text
	plt.figtext(0.84, 0.99, version_name, size=8)

	return;

def tuteta_open_in_mr(sample_mdate, sample_id, sample_no):

	file_xy = sample_mdate + sample_id + '-xy' 
	file_xz = sample_mdate + sample_id + '-xz'
	file_yz = sample_mdate + sample_id + '-yz'
	
	# load raw data from text file, skiprows=1 --> gets rid of the rxyz file headder
	x1, y1 = np.loadtxt(file_xy, skiprows=3, usecols=(0,2), unpack=True)
	x2, y2 = np.loadtxt(file_xz, skiprows=3, usecols=(0,1),unpack=True)
	x3, y3 = np.loadtxt(file_yz, skiprows=3, usecols=(0,2), unpack=True)

	return np.array([x1,y1,x2,y2,x3,y3]);

def tuteta_open_in_hv(sample_mdate, sample_id, sample_no):

	file_xy = sample_mdate + sample_id + '-xy_vh' 
	file_xz = sample_mdate + sample_id + '-xz_vh'
	file_yz = sample_mdate + sample_id + '-yz_vh'
	
	# load raw data from text file, skiprows=1 --> gets rid of the rxyz file headder
	x1, y1 = np.loadtxt(file_xy, skiprows=3, usecols=(0,2), unpack=True)
	x2, y2 = np.loadtxt(file_xz, skiprows=3, usecols=(0,2),unpack=True)
	x3, y3 = np.loadtxt(file_yz, skiprows=3, usecols=(0,2), unpack=True)

	return np.array([x1,y1,x2,y2,x3,y3]);

def tuteta_shift(raw_tuteta_data, shift_xy, shift_xz, shift_yz):

	x1 = raw_tuteta_data[0] * (xmax / np.amax(raw_tuteta_data[0]))
	x2 = raw_tuteta_data[2] * (xmax / np.amax(raw_tuteta_data[2]))
	x3 = raw_tuteta_data[4] * (xmax / np.amax(raw_tuteta_data[4]))

	y1 = raw_tuteta_data[1] - shift_xy
	y2 = raw_tuteta_data[3] - shift_xz
	y3 = raw_tuteta_data[5] - shift_yz

	return np.array([x1,y1,x2,y2,x3,y3]);

def tuteta_fit_mr(R0, A1, A2, d1, d2):

	x = np.arange(0, 2*np.pi, 0.1)

	R = R0 + A1*np.cos(x+d1) + A2*np.cos(2*(x+d2))
	tx1, = plt.plot(x*180/np.pi, R, 'b--', label=label_xy_fit)

	return;

def tuteta_resit_cal(tuteta_data):

	# look for the peak value in this area surrounding 0,90,180,270,360 deg point
	deg_spread = 15
	# index of the x values in the input data
	indx = 0

	a_range = np.where( (tuteta_data[indx] >= 0) & (tuteta_data[indx] <=  deg_spread) )
	b_range = np.where( (tuteta_data[indx] >= np.int_(0.25*xmax) - deg_spread) & (tuteta_data[indx] <= np.int_(0.25*xmax) + deg_spread) )
	c_range = np.where( (tuteta_data[indx] >= np.int_(0.5*xmax) - deg_spread) & (tuteta_data[indx] <= np.int_(0.5*xmax) + deg_spread) )
	d_range = np.where( (tuteta_data[indx] >= np.int_(0.75*xmax) - deg_spread) & (tuteta_data[indx] <= np.int_(0.75*xmax) + deg_spread) )
	e_range = np.where( (tuteta_data[indx] >= np.int_(xmax) - deg_spread) & (tuteta_data[indx] <= np.int_(xmax) ) )
		
	R_xy_000 = np.average(tuteta_data[1][a_range[0][0]:a_range[0][np.size(a_range)-1]])
	R_xy_090 = np.average(tuteta_data[1][b_range[0][0]:b_range[0][np.size(b_range)-1]])

	R_xz_000 = np.average(tuteta_data[3][a_range[0][0]:a_range[0][np.size(a_range)-1]])
	R_xz_090 = np.average(tuteta_data[3][b_range[0][0]:b_range[0][np.size(b_range)-1]])

	R_yz_000 = np.average(tuteta_data[5][a_range[0][0]:a_range[0][np.size(a_range)-1]])
	R_yz_090 = np.average(tuteta_data[5][b_range[0][0]:b_range[0][np.size(b_range)-1]])
	
	if R_xy_000 > R_xy_090:

		R_xy_000deg = np.amax(tuteta_data[1][a_range[0][0]:a_range[0][np.size(a_range)-1]])
		R_xy_090deg = np.amin(tuteta_data[1][b_range[0][0]:b_range[0][np.size(b_range)-1]])
		R_xy_180deg = np.amax(tuteta_data[1][c_range[0][0]:c_range[0][np.size(c_range)-1]])
		R_xy_270deg = np.amin(tuteta_data[1][d_range[0][0]:d_range[0][np.size(d_range)-1]])
		R_xy_360deg = np.amax(tuteta_data[1][e_range[0][0]:e_range[0][np.size(e_range)-1]])

		R_xy_min = 0.5*(R_xy_090deg + R_xy_270deg)
		R_xy_max = (R_xy_000deg + R_xy_180deg + R_xy_360deg) / 3

	elif R_xy_000 < R_xy_090:
		
		R_xy_000deg = np.amin(tuteta_data[1][a_range[0][0]:a_range[0][np.size(a_range)-1]])
		R_xy_090deg = np.amax(tuteta_data[1][b_range[0][0]:b_range[0][np.size(b_range)-1]])
		R_xy_180deg = np.amin(tuteta_data[1][c_range[0][0]:c_range[0][np.size(c_range)-1]])
		R_xy_270deg = np.amax(tuteta_data[1][d_range[0][0]:d_range[0][np.size(d_range)-1]])
		R_xy_360deg = np.amin(tuteta_data[1][e_range[0][0]:e_range[0][np.size(e_range)-1]])

		R_xy_max = 0.5*(R_xy_090deg + R_xy_270deg)
		R_xy_min = (R_xy_000deg + R_xy_180deg + R_xy_360deg) / 3

	if R_xz_000 > R_xz_090:

		R_xz_000deg = np.amax(tuteta_data[3][a_range[0][0]:a_range[0][np.size(a_range)-1]])
		R_xz_090deg = np.amin(tuteta_data[3][b_range[0][0]:b_range[0][np.size(b_range)-1]])
		R_xz_180deg = np.amax(tuteta_data[3][c_range[0][0]:c_range[0][np.size(c_range)-1]])
		R_xz_270deg = np.amin(tuteta_data[3][d_range[0][0]:d_range[0][np.size(d_range)-1]])
		R_xz_360deg = np.amax(tuteta_data[3][e_range[0][0]:e_range[0][np.size(e_range)-1]])

		R_xz_min = 0.5*(R_xz_090deg + R_xz_270deg)
		R_xz_max = (R_xz_000deg + R_xz_180deg + R_xz_360deg) / 3

	elif R_xz_000 < R_xz_090:

		R_xz_000deg = np.amin(tuteta_data[3][a_range[0][0]:a_range[0][np.size(a_range)-1]])
		R_xz_090deg = np.amax(tuteta_data[3][b_range[0][0]:b_range[0][np.size(b_range)-1]])
		R_xz_180deg = np.amin(tuteta_data[3][c_range[0][0]:c_range[0][np.size(c_range)-1]])
		R_xz_270deg = np.amax(tuteta_data[3][d_range[0][0]:d_range[0][np.size(d_range)-1]])
		R_xz_360deg = np.amin(tuteta_data[3][e_range[0][0]:e_range[0][np.size(e_range)-1]])

		R_xz_max = 0.5*(R_xz_090deg + R_xz_270deg)
		R_xz_min = (R_xz_000deg + R_xz_180deg + R_xz_360deg) / 3

	if R_yz_000 > R_yz_090:

		R_yz_000deg = np.amax(tuteta_data[5][a_range[0][0]:a_range[0][np.size(a_range)-1]])
		R_yz_090deg = np.amin(tuteta_data[5][b_range[0][0]:b_range[0][np.size(b_range)-1]])
		R_yz_180deg = np.amax(tuteta_data[5][c_range[0][0]:c_range[0][np.size(c_range)-1]])
		R_yz_270deg = np.amin(tuteta_data[5][d_range[0][0]:d_range[0][np.size(d_range)-1]])
		R_yz_360deg = np.amax(tuteta_data[5][e_range[0][0]:e_range[0][np.size(e_range)-1]])

		R_yz_min = 0.5*(R_yz_090deg + R_yz_270deg)
		R_yz_max = (R_yz_000deg + R_yz_180deg + R_yz_360deg) / 3

	elif R_yz_000 < R_yz_090:

		R_yz_000deg = np.amin(tuteta_data[5][a_range[0][0]:a_range[0][np.size(a_range)-1]])
		R_yz_090deg = np.amax(tuteta_data[5][b_range[0][0]:b_range[0][np.size(b_range)-1]])
		R_yz_180deg = np.amin(tuteta_data[5][c_range[0][0]:c_range[0][np.size(c_range)-1]])
		R_yz_270deg = np.amax(tuteta_data[5][d_range[0][0]:d_range[0][np.size(d_range)-1]])
		R_yz_360deg = np.amin(tuteta_data[5][e_range[0][0]:e_range[0][np.size(e_range)-1]])

		R_yz_max = 0.5*(R_yz_090deg + R_yz_270deg)
		R_yz_min = (R_yz_000deg + R_yz_180deg + R_yz_360deg) / 3	

	AMR_ratio_xy = 100*(R_xy_max - R_xy_min) / R_xy_min
	AMR_ratio_xz = 100*(R_xz_max - R_xz_min) / R_xz_min
	SMR_ratio_yz = 100*(R_yz_max - R_yz_min) / R_yz_min

	print(R_xy_000deg / sample_current, R_xy_090deg / sample_current, R_xy_180deg / sample_current, R_xy_270deg / sample_current, R_xy_360deg / sample_current)
	print(R_xz_000deg / sample_current, R_xz_090deg / sample_current, R_xz_180deg / sample_current, R_xz_270deg / sample_current, R_xz_360deg / sample_current)
	print(R_yz_000deg / sample_current, R_yz_090deg / sample_current, R_yz_180deg / sample_current, R_yz_270deg / sample_current, R_yz_360deg / sample_current)
	
	print('AMR ratio xy : ' + np.str('% 1.3f' % AMR_ratio_xy) + '%')
	print('AMR ratio xz : ' + np.str('% 1.3f' % AMR_ratio_xz) + '%')
	print('SMR ratio yz : ' + np.str('% 1.3f' % SMR_ratio_yz) + '%')

	return np.array([(R_xy_000deg / sample_current, R_xy_090deg / sample_current, R_xy_180deg / sample_current, R_xy_270deg / sample_current, R_xy_360deg / sample_current, AMR_ratio_xy),(R_xz_000deg / sample_current, R_xz_090deg / sample_current, R_xz_180deg / sample_current, R_xz_270deg / sample_current, R_xz_360deg / sample_current, AMR_ratio_xz),(R_yz_000deg / sample_current, R_yz_090deg / sample_current, R_yz_180deg / sample_current, R_yz_270deg / sample_current, R_yz_360deg / sample_current, SMR_ratio_yz)]);

def tuteta_plot(tuteta_data, sample_current, ymin, ymax, label_xy, label_xz, label_yz, polarity_xy, polarity_xz, polarity_yz):
	# plot recalculated data with background removed
	# 'ro-' -> red circles with solid line
	tx1, = plt.plot(tuteta_data[0], polarity_xy * tuteta_data[1] / sample_current, 'co', mfc='lightcyan', markersize=6, label=label_xy)
	tx2, = plt.plot(tuteta_data[2], polarity_xz * tuteta_data[3] / sample_current, 'mo', mfc='lavenderblush', markersize=6, label=label_xz)
	tx3, = plt.plot(tuteta_data[4], polarity_yz * tuteta_data[5] / sample_current, 'yo', mfc='lightyellow', markersize=6, label=label_yz)
	
	# display the legend for the defined labels
	plt.legend([tx1, tx2, tx3], [label_xy, label_xz, label_yz], loc='upper left', frameon=False)
	
	# set x,y limits
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)

	return;

# Create a new figure of size 8x8 inches, using 100 dots per inch
# protip: A4 is 8.3 x 11.7 inches
fig = plt.figure(figsize=(10, 14), dpi=72)
spec = gridspec.GridSpec(ncols=3, nrows=3)
fig.canvas.set_window_title('mrhv_plots_' + sample_mdate + sample_id)

# set the sample no for which to calculate and plot data
sample_no = 0
# set title of the plot 
#plt_title = gen_plot_title(sample_no)
# load raw data from measuremnt files for sample_no: 3 
tuteta_mr_files_in = tuteta_open_in_mr(sample_mdate, sample_id, sample_no)
tuteta_hv_files_in = tuteta_open_in_hv(sample_mdate, sample_id, sample_no)
# shift the data vertically 
tuteta_shifted_mr = tuteta_shift(tuteta_mr_files_in, mr_shift_xy, mr_shift_xz, mr_shift_yz)
tuteta_shifted_hv = tuteta_shift(tuteta_hv_files_in, hv_shift_xy, hv_shift_xz, hv_shift_yz)

# calcluate MR ratios for the MR curves
calculated_resist = tuteta_resit_cal(tuteta_shifted_mr)

# add subplot of first sample data (mr curves)
xy1 = fig.add_subplot(6,6,(1,24))
# plot the data for xy, xz, yz mr curves
tuteta_plot(tuteta_shifted_mr, sample_current, mr_ymin, mr_ymax, mr_label_xy, mr_label_xz, mr_label_yz, 1.0, 1.0, 1.0)
custom_axis_formater(axis_label_theta, axis_label_ohm, xmin, xmax, mr_ymin, mr_ymax, xprec, yprec)
set_plot_title(1)
# plot the data for xy, xz, yz hv curves
xy2 = fig.add_subplot(6,6,(28,36))
tuteta_plot(tuteta_shifted_hv, sample_current, hv_ymin, hv_ymax, hv_label_xy, hv_label_xz, hv_label_yz, 1.0, 1.0, 1.0)
plt.legend(loc='upper left', ncol=3, frameon=False)
custom_axis_formater(axis_label_theta, axis_label_ohm, xmin, xmax, hv_ymin, hv_ymax, xprec, yprec)
set_plot_title(2)
# generate plot description containing results summary 
tuteta_gen_desc(sample_field, sample_current, calculated_resist)

# save the calculated R peak values, AMR, SMR to a textfile
#np.savetxt(out_tsv, np.c_[calculated_resist[0], calculated_resist[1], calculated_resist[2]], delimiter='	' , header=' generated using: ' + version_name + '\n\n' + ' input dataset: ' + sample_mdate + sample_id + '\n\n' + ' Rxy[Ohm]	Rxz[Ohm]	Ryz[Ohm]', footer=' AMR_ratio(xy)[%]	AMR_ratio(xz)[%]	SMR_ratio(yz)[%]', comments='#')   # X is an array
# fit the xy curve
#tuteta_fit_mr(R0, A1, A2, d1, d2)
# format axis and add labels


fig.tight_layout(pad=5.0, w_pad=0.0, h_pad=0.0)
#plt.subplots_adjust(left=0.15, bottom=0.3, wspace=0.4, hspace=0.2)

# write a pdf file with fig and close
pp = PdfPages(out_pdf)
pp.savefig(fig)
pp.close()

# save as .svg too
fig = plt.savefig(out_svg)

# show plot preview
plt.show()