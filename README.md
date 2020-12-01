# Human-Cochlear-Model
This repo provides a Python code for simulating the basilar membrane motion in the 1D and 2D transmission line models of the cochlea without a middle ear model. Input waveform is NOT sound pressure, and is the stapes velocity. Amplitude of input is ad-hoc adjusted by the wav-file loader to fit compressive nonlinearity. Sampling frequency in an input wav-file can be set arbitrary. However, this value in an output from the model fixed at 200 kHz.

# Demonstration
To run the 1D model or the 2D model, execute CochlearModel_1D_Direct.py or CochlearModel_2D_Direct.py, respectively. This demo calculate the BM responses for 0.25, 1, 4 kHz tones when input level varied from 0 to 100 dB within 20 dB step. Duration of the pure tone is 100 msec.
