import os
import numpy as np
import matplotlib.pyplot as plt


RESULTS_FOLDER = './results/'
NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
K_IN_M = 1000.0
REBUF_P = 4.3
SMOOTH_P = 1
COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired 
SIM_DP = 'sim_dp'

SCHEMES = ['sim_bb', 'sim_mpc', 'sim_rl', 'sim_dp']

# br - bitrate utility, rp - rebuffering penalty, sp - smoothness penalty
def qoe_calc_lin(R_n, T_n):
	br = sum(R_n)
	rp = 4.3 * sum(T_n)
	sp = sum([abs(R_n[i+1] - R_n[i]) for i in range(len(R_n)-1)])
	qoe_res =  br - rp - sp
	return qoe_res/len(R_n), br, rp, sp

def qoe_calc_log(R_n, T_n):
	R_n = np.array(R_n)
	br = sum(np.log(R_n/np.amin(R_n)))
	rp = 2.66 * sum(T_n)
	sp = sum([abs(np.log(R_n[i+1]/np.amin(R_n)) - np.log(R_n[i]/np.amin(R_n))) for i in range(len(R_n)-1)])
	qoe_res =  br - rp - sp
	return qoe_res/len(R_n), br, rp, sp

def qoe_calc_hd(R_n, T_n):
	qR = {0.3 : 1, 0.75 : 2, 1.2 : 3, 1.85 : 12, 2.85 : 15, 4.3 : 20}
	br = sum([qR[R_n[i]] for i in range(len(R_n))])
	rp = 8 * sum(T_n)
	sp = sum([abs(qR[R_n[i+1]] - qR[R_n[i]]) for i in range(len(R_n)-1)])
	qoe_res =  br - rp - sp
	return qoe_res/len(R_n), br, rp, sp

def main():
	time_all = {}
	bit_rate_all = {}
	buff_all = {}
	bw_all = {}
	raw_reward_all = {}

	for scheme in SCHEMES:
		time_all[scheme] = {}
		raw_reward_all[scheme] = []
		bit_rate_all[scheme] = {}
		buff_all[scheme] = {}
		bw_all[scheme] = {}

	qoes = {'sim_dp' : {'lin' : [], 'log' : [], 'hd' : []},
			'sim_bb' : {'lin' : [], 'log' : [], 'hd' : []},
			'sim_rl' : {'lin' : [], 'log' : [], 'hd' : []},
			'sim_mpc': {'lin' : [], 'log' : [], 'hd' : []}}

	log_files = os.listdir(RESULTS_FOLDER)
	for log_file in log_files:

		time_ms = []
		bit_rate = []
		buff = []
		bw = []
		reward = []

		br = []
		rp = []

		#print(log_file)
		
		if SIM_DP in log_file:
			with open(RESULTS_FOLDER + log_file, 'rb') as f:
				last_t = 0
				last_b = 0
				last_q = 1
				lines = []
				for line in f:
					lines.append(line)
					parse = line.split()
					if len(parse) >= 6:
						time_ms.append(float(parse[3]))
						bit_rate.append(VIDEO_BIT_RATE[int(parse[6])])
						buff.append(float(parse[4]))
						bw.append(float(parse[5]))
				
				for line in reversed(lines):
					parse = line.split()
					r = 0
					if len(parse) > 1:
						t = float(parse[3])
						b = float(parse[4])
						q = int(parse[6])
						if b == 4:
							rebuff = (t - last_t) - last_b
							assert rebuff >= -1e-2
							r -= REBUF_P * rebuff

						r += VIDEO_BIT_RATE[q] / K_IN_M
						r -= SMOOTH_P * np.abs(VIDEO_BIT_RATE[q] - VIDEO_BIT_RATE[last_q]) / K_IN_M
						reward.append(r)

						br.append(VIDEO_BIT_RATE[q] / K_IN_M)
						rp.append((t - last_t) - last_b)

						last_t = t
						last_b = b
						last_q = q

			time_ms = time_ms[::-1]
			bit_rate = bit_rate[::-1]
			buff = buff[::-1]
			bw = bw[::-1]

			#qoe_lin, brlin, rplin, splin = qoe_calc_lin(br,rp)
			qoe_lin, brlin, rplin, splin = qoe_calc_lin(br,np.zeros(len(br)))
			qoe_log, brlog, rplog, splog = qoe_calc_log(br,np.zeros(len(br)))
			qoe_hd, brhd, rphd, sphd = qoe_calc_hd(br,np.zeros(len(br)))
			#print(qoe_lin, qoe_log, qoe_hd)
			qoes['sim_dp']['lin'].append([qoe_lin])
			qoes['sim_dp']['log'].append([qoe_log])
			qoes['sim_dp']['hd'].append([qoe_hd])
			print("\n",np.mean(br),np.mean(rp))

		else:
			with open(RESULTS_FOLDER + log_file, 'r') as f:
				all_lines = f.readlines()
				for line in all_lines:
					parse = line.split()
					if len(parse) <= 1:
						break
					if len(parse) != 4:
						time_ms.append(float(parse[0]))
						bit_rate.append(int(parse[1]))
						buff.append(float(parse[2]))
						bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
						reward.append(float(parse[6]))

						br.append(int(parse[1])/K_IN_M)
						rp.append(float(parse[3]))
				
				for scheme in SCHEMES:
					if scheme in log_file:
						current_scheme = scheme

				qoes[current_scheme]['lin'].append(qoe_calc_lin(br, rp))
				qoes[current_scheme]['log'].append(qoe_calc_log(br, rp))
				qoes[current_scheme]['hd'].append(qoe_calc_hd(br, rp))				

				
		time_ms = np.array(time_ms)
		time_ms -= time_ms[0]
		
		# print log_file

		for scheme in SCHEMES:
			if scheme in log_file:
				time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
				bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
				buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
				bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
				break

	labels = ["QoE (lin)", "QoE (log)", "QoE (hd)"]
	sublabels = SCHEMES.copy()
	sublabels.remove('sim_dp')
	metrics = ['lin', 'log', 'hd']

	# ---- ---- ---- ----
	# Normalized Average QoE Graphs
	# ---- ---- ---- ----
	#"""
	lin_qoe_res = []
	log_qoe_res = []
	hd_qoe_res = []

	results = {'sim_bb' : [], 'sim_mpc' : [], 'sim_rl' : []}

	for scheme in sublabels:
		for mt in metrics:
			results[scheme].append(np.mean([i[0] for i in qoes[scheme][mt][:]]))

	for scheme in sublabels:
		for i in range(len(metrics)):
			results[scheme][i] /= results['sim_rl'][i]

	x_pos = np.arange(len(labels))
	bar_width = 0.2

	fig, ax = plt.subplots()
	for (atr1, measurement), i in zip(results.items(), range(len(sublabels))):
		ax.bar(x_pos + i * bar_width, measurement, bar_width, label = sublabels[i])
	
	ax.set_ylabel('Normalized Average QoE')
	ax.set_xticks(x_pos + bar_width)
	ax.set_xticklabels(labels)
	ax.legend()
	
	plt.show()
	#"""
	# ---- ---- ---- ----
	# Average QoE Graphs
	# ---- ---- ---- ----
	#"""
	fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
	plt.subplots_adjust(wspace=0.5)

	for ax, mt, i, rng in zip(axs.ravel(), metrics, range(len(metrics)), [[-0.5, 3, 3.5, 0.5], [-0.5, 2.5, 3, 0.5], [-1, 14, 15, 3]]):
		for scheme in SCHEMES:

			qoe_calc = sorted([i[0] for i in qoes[scheme][mt][:]])
			if scheme == 'sim_dp':
				qoe_calc = (np.array(qoe_calc)).tolist()
				#print(np.mean(qoe_calc))
			#else :
				#print(scheme,np.mean(qoe_calc))
			x_axis = np.arange(rng[0], rng[1], rng[2]/len(qoe_calc))
			ax.plot(x_axis, [sum(x_axis[i] >= qoe_calc)/len(qoe_calc) for i in range(len(qoe_calc))])

		ax.legend(SCHEMES)
		ax.set_ylabel('CDF')
		ax.set_xlabel('Avg QoE (' + metrics[i] + ')')
		ax.set_xticks(np.arange(rng[0], rng[1]+1, (rng[-1])))

	plt.show()
	#"""
	# ---- ---- ---- ----
	# QoE Component Graphs
	# ---- ---- ---- ----
	#"""
	fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(4, 15))
	plt.subplots_adjust(hspace=0.5)

	labels = ["Bitrate Utility", "Rebuffering Penalty", "Smoothness Penalty"]

	for ax, mt in zip(axs.ravel(), metrics):
		br_qoes = []
		rp_qoes = []
		sp_qoes = []

		results = {'sim_bb' : [], 'sim_mpc' : [], 'sim_rl' : []}

		for scheme in sublabels:
			results[scheme].append([np.mean([i[1] for i in qoes[scheme][mt][:]]),
									np.mean([i[2] for i in qoes[scheme][mt][:]]),
									np.mean([i[3] for i in qoes[scheme][mt][:]])])
	
		x_pos = np.arange(len(labels))
		bar_width = 0.2

		for (atr1, measurement), i in zip(results.items(), range(len(sublabels))):
			ax.bar(x_pos + i * bar_width, np.array(measurement[0])/VIDEO_LEN, bar_width, label = sublabels[i])

		ax.set_ylabel('Average Value')
		ax.set_title('QoE (' + mt + ')')
		ax.set_xticks(x_pos + bar_width)
		ax.set_xticklabels(labels)
		ax.legend()

	plt.show()
	#"""
if __name__ == '__main__':
	main()
