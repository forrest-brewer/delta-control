# Matlab Code
This is the transfer function from the matlab code. It's a low pass filter with a bandwidth of 2kHz.

![](../../dIIR_filter_sim_matlab/lp_filter_2k/tf.png)

The states.

![](../../dIIR_filter_sim_matlab/lp_filter_2k/state.png)


# $\Sigma\Delta$ Modulator Input
A series of tones was encoded with a 2nd order $\Sigma\Delta$ modulator.


![](sd_psd_in.png)

Here is the time domain plot. Something weird is going on at the beginning but then it settles out. Are the initial conditions wrong?

![](sd_in.png)

# $\Sigma\Delta$ Filter with Tones Inputted

Below is are the states from the $\Sigma\Delta$ filter.

![](states_subplot.png)

on one plot.


![](states_all.png)


The last state, 3, is much larger than the others. Below a repeat of the plot minus state 3. The matlab code indicated that all the state were about the same order of magnitude. Why is the filter different? Why is state 3 an order of magnitude larger?


![](states_most.png)


Here is the PSD of the output from the $\Sigma\Delta$ filter. It sort of looks like a 2kHz LP filter.

![](sd_psd_out.png)

# $\Sigma\Delta$ Filter White Noise
A random signal was generated then low pass filtered at 22050Hz, Nyquist. Here is the PSD of the $\Sigma\Delta$ encoded signal.

![](wn_in_psd.png)

Here is the PSD from the $\Sigma\Delta$ filter. It doesn't look like a low 2kHz pass filter. 

![](wn_out_psd.png)