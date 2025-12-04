# PS-analysis
read and format sampled data of bus voltage and branch current from a laboratory-scale 5 bus system with RLC load, synchronous generation, transmission lines and transformers.

## content
### [/data](data)
    contains the .txt files from laboratory measurements
### [/images](images)
    contains the network images exported with current and power as individual files, for each case in /data
### [/summaries](summaries)
    contains the summary in table formatfor each case in /data
### [graph.py](graph.py)
    helper function to generate graph-type plots, accept calculated quantities as argments, this function does not perform electrical calculations
### [helpers.py](helpers.py)
    general helper functions used across other .py files. Contains data handling, electrical calculations, integration methods and waveforms figure of merit calculation
### [main.py](main.py)
    call other modules in the correct order to compute the experiment calculations. its possible to set the experiments to be performed with the `experiment` argument of the main() function.

## reproducibility
at [line 338](main.py) of main.py define which experiments to perform. by default, all the ones from [/data](/data) are included there. run main.py

## approach
### phase angle calculation
Take the Fourier transform


$$ w_{1,fft}=FFT(wave_1) $$
$$ w_{2,fft}=FFT(wave_2) $$

Find the fundamental phase (assume it’s the highest peak)
$$ w_{1,angle}=angle(wave_{1,fundamental}) $$
$$ w_{2,angle}=angle(wave_{2,fundamental}) $$

Compute the phase difference between two waves
$$ \phi_{2,1}=w_{2,angle}-w_{1,angle} $$

Assuming Bus1 voltage as the reference (angle = 0), for all the other voltages and current waveforms, phase angle will be computed as the difference between the waveform of interest to the waveform of Bus1 voltage, therefore, the calculated phase angle will be the actual phase angle of the waveform.

### waveform average value
$$ W_{avg}=\frac{1}{T}\int_0^T{w(t)dt} $$
### waveform rms value
$$ W_{rms}=\sqrt{\frac{1}{T}\int_0^T{w^2(t)dt}} $$
### active power calculation
$$ P_{1\phi}=P_{avg}=\frac{1}{T}\int_0^T{v(t)\cdot i(t) \, dt} $$
$$ P_{3\phi}=3\, P_{1\phi} $$
assuming balanced system.
### apparent power calculation
$$ S_{1\phi}=V_{rms}\, I_{rms} $$
$$ S_{3\phi}=3\, S_{1\phi} $$
assuming balanced system.
### reactive power calculation
$$ Q_{1\phi}=P_{1\phi}\, \tan{\phi_{v,i}} $$
$$ \phi_{v,i}=\phi_{v}-\phi_{i} $$
$$ Q_{3\phi}=3\, Q_{1\phi} $$
assuming balanced system.

## results
as described in the report (TBI)