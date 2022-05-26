# EEG preprocessing for [**DEAP**](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html) dataset

This repo is based on DEAP, which is among the most-cited datasets for EEG-based emotion recognition.
You can find scripts to a) download DEAP, b) preprocess its EEG signals and c) perform feature extraction (Power Spectral Density, Differential Entropy, Power Spectral Density asymmetry, Differential Entropy asymmetry).

Note that, to download DEAP dataset using this code, you need to create a JSON file containing your credentials, using the `create_json.py` file.

After downloading the dataset in BDF file format, you can proceed to preprocessing.
The pipeline for EEG preprocessing is based on the [**well-known  steps**](https://erpinfo.org/order-of-steps) of Steve Luck.
More specifically, the preprocessing script performs the following steps:
1. Load .bdf file for each subject
2. Get channel names (1-32 are EEG, 33-40 are EMG/EOG, the last one [48/49] is Stimuli status)
3. Drop non-EEG channels (keep stimulus/status channel)
4. Set montage of electrode locations to Biosemi32
5. Get sampling frequency
6. Apply filters on data: notch filter @ 50Hz, bandpass filter @ 4-45Hz
7. Re-reference EEG channels to the common average reference
8. Get events from stimulus/status channel
9. Keep only trial-start events (event ID == 4 is the stimulus onset, i.e. the beginning of each trial)
10. Epoch data, using the trial-start events, and setting tmin=-5.0 and tmax=+60.0
11. Define an ICA transformation, using FastICA and 32 channels
12. Fit the ICA to the epoched data
	12a: Plot ICA sources (optional)
	12b: Save ICA sources as figures (optional)
13. Plot ICA components and manually reject eye-movement related components
	13a: Save ICA components as figures
14. Apply the fitted ICA to the epoched data, to obtain the cleaned epoched data
15. Plot the PSD of the clean epoched data
16. Downsample the frequency of the cleaned epoched data, from 512Hz to 128Hz
17. Get the EEG time-series of the downsampled cleaned epoched data
18. Re-order the EEG channels, to follow the Geneva order
19. Re-order the trials, to follow the experiment_ID order (same video stimulus order)


----------------------------------


You can see an example of a subject's ICA components here:

<p>
  <img src="https://raw.githubusercontent.com/gzoumpourlis/DEAP_MNE_preprocessing/main/figures/s01_ICA_components.png" width="800" title="ICA components of subject 01 (notice the rejected component ICA017 with grey font color)">
</p>

You can see an example of a subject's PSD plot here:

<p>
  <img src="https://raw.githubusercontent.com/gzoumpourlis/DEAP_MNE_preprocessing/main/figures/s16_PSD_downsampled.png" width="800" title="PSD plot of subject 16">
</p>
