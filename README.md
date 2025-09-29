<h3>
Spectrogram Cross Correlation (SPCC) for Synthetic Speech Classification
</h3>
<p>
This is a web application written in Go that make use of the html/template package to dynamically
create the web page. Start the web server at bin\specXcorrspeech.exe and connect to it from your web browser
at http://127.0.0.1:8080/speechSPCCtest. The program creates synthetic speech and generates spectrograms from them. 
Spectrograms are 3-dimensional plots of the time-frequency content of the audio. The spectral power at a 
particular frequency and time is shown as a grayscale color, with black having the greatest power and white 
having the least. Short-time Fourier transforms (STFT) are used in 32 ms time frames with no overlap. 
between FFTs. Time domain plots of the audio waveform can be displayed as well as the spectrogram. 
The user can generate new synthetic speech of 2.048 seconds (16,384 samples) by selecting the New Speech radio button.  
Upon selecting the <i>Submit</i> button, the user will hear the speech on their computer's audio device. 
To hear the synthetic speech requires the <b>fmedia</b> program to be in the PATH environmental variable.
A WAV file is created from the synthetic speach and submitted to the fmedia program to be played.
</p>

<p>
The FFT size is 256. The sampling rate is 8,000 Hz which produces a Nyquist critical frequency of 4,000 Hz. 
The grayscale consists of five colors. As stated above, black RGB(0,0,0), has the greatest power in the 
Power Spectral Density (PSD) bin.  There are 256 bins in the 256-pt FFT and each bin is 8,000/256 = 31.25 Hz wide.
</p>

<p>
The synthetic speech is generated using a sum of sinusoids (voiced) or gaussian noise (unvoiced) in
20-30 ms frames.  If voiced, the fundamental frequency is randomly chosen from between 200 and 800 Hz. Each voiced
speech has 1-5 subfrequencies with a smaller amplitude than the fundamental.  The amplitudes are randomly
chosen and can be varied.  The duration of each frame can also be varied.  The variation of these parameters
will test the generalization capabilities of the Neural Network.  The testing phase varies the parameters
based upon the user input.  The percentage of correct classification is presented in graphical and tabular
forms upon completion of the testing.
</p>

<p>
Spectrograms of the synthetic speech are created as references to be cross correlated against the test 
synthetic speech spectrograms.  The references do not have any variation in frequency, amplitude, or frame duration.
The test speech samples do have variation in these parameters.  The user determines how much variation to apply.
As the test synthetic speech spectrograms are generated, they are cross correlated against each reference.  The maximum
cross correlation is declared to be the pattern and is tabulated for count and correctness.  The cross correlations
are done with goroutines; each spectrogram reference has a goroutine with which it does the cross correlation.  Machines
with more logical processors will benefit from the greater concurrency.  A result channel is used by the goroutines to
send their cross correlation results back to the collector goroutine.  The collector goroutine will then determine which
reference has the greatest cross correlation.  It will be declared correct if it matches the test spectrogram.
</p>

<h4>Spectorgram Cross Correlation Test Results</h4>
<img width="1098" height="1007" alt="image" src="https://github.com/user-attachments/assets/cdc3d7b4-56bc-4735-bd1c-1fe72c0063c8" />
<h4>Synthetic Speech Display Time Domain, Speech Pattern 1</h4>
<img width="1314" height="896" alt="image" src="https://github.com/user-attachments/assets/75141e35-0924-4d4e-b56a-cfeae3829e30" />
<h4>Synthetic Speech Display Spectrogram, Speech Pattern 1</h4>
<img width="1316" height="909" alt="image" src="https://github.com/user-attachments/assets/f63c7320-74ee-4820-9216-f8ee157a7bb8" />

