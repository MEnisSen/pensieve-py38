# Note:
Any test done in this section will be the result of the model configured with Linear QoE. To reproduce the figures (7, 8, 9, 10) from the paper, be sure to also train the model for Logarithmic QoE and HD QoE. Changes were made as the original `plot_results.py` code file didn't produce the mentioned figures from the paper.

*_Each simulation should be configured for each QoE metric!_*

# Readme

Make sure actual video files are stored in `video_server/video[1-6]`, then run
```
python get_video_sizes
```

Put testing data in `test/cooked_traces`. The trace format for simulation is `[time_stamp (sec), throughput (Mbit/sec)]`. Sample testing data can be downloaded from `test_sim_traces` in https://www.dropbox.com/sh/ss0zs1lc4cklu3u/AAB-8WC3cHD4PTtYT0E4M19Ja?dl=0. More details of data preparation can be found in `traces/`.

Trained RL model needs to be saved in `test/models/`. We provided a sample pretrained model with linear QoE as the reward signal. It can be loaded by setting `NN_MODEL = './models/pretrain_linear_reward.ckpt'` in `rl_no_training.py`.

To test a trained model, run 
```
python rl_no_training.py
```

Results will be saved in `test/results/`. Similarly, one can also run `bb.py` for buffer-based simulation, `mpc.py` for robustMPC simulation, and `dp.cc` for offline optimal simulation.

To view the results, modify `SCHEMES` in `plot_graphs.py` (it checks the file name of the log and matches to the corresponding ABR algorithm), then run 
```
python plot_graphs.py
```
