/*
Spectrogram Cross-Correlation for synthetic speech recognition
This is a web application that uses the html/template package to create the HTML.
The URL is http://127.0.0.1:8080/specXcorr.  A cross-correlation is performed between
pattern templates and the trial samples.  The sample's parameters, amplitude, pitch, and
frame period, are varied in order to simulate the human speech characteristics.

This application classifies synthetic speech patterns.  The spectrogram of
each speech pattern file is calculated and the spectrogram is the input to the SPCC.
The SPCC classifies the speech pattern based on its spectral content versus time. The test
results are shown.  The user can plot the time domain or the spectrogram
(frequency versus time) of the synthetic speech.  The spectrogram is a three-dimentional
plot of the spectral power versus time.  The third dimension is a grayscale color.
Short-time Fourier Transforms (STFT) are used to compute the FFT from 20-30 ms blocks
of synthetic speech data.

The synthetic speech is generated with a sum of sinusoids (voiced) or gaussian noise (unvoiced) in
20-30 ms frames.  If voiced, the fundamental is randomly chosen from between 200 and 800 Hz. Each voiced
speech has 1-5 subfrequencies with a smaller amplitude than the fundamental.  The amplitudes are randomly
chosen and can be varied.  The duration of each frame can also be varied.  The variation of these parameters
will test the generalization capabilities of the Neural Network.  The testing phase varies the parameters
based upon the user input.  The percentage of correct classification is presented in graphical and tabular
forms upon completion of the testing.
*/

package main

import (
	"bufio"
	"fmt"
	"html/template"
	"log"
	"math"
	"math/cmplx"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/mjibson/go-dsp/fft"
)

const (
	addr               = "127.0.0.1:8080"             // http server listen address
	fileTestingSPCC    = "templates/testingSPCC.html" // html for testing SPCC
	fileDisplaySPCC    = "templates/displaySPCC.html" // html for speech time or spectrogram plots
	patternTestingSPCC = "/speechSPCCtest"            // http handler for testing the SPCC
	patternDisplaySPCC = "/speechSPCCdisplay"         // http handler for displaying the SPCC
	xlabels            = 11                           // # labels on x axis
	ylabels            = 11                           // # labels on y axis
	synSpeech          = "synSpeech.wav"              // synthetic speech wav file
	dataDir            = "data/"                      // directory for the weights and synthetic speech files
	rows               = 300                          // rows in canvas
	cols               = 300                          // columns in canvas
	sampleRate         = 8000                         // Hz or samples/sec
	bitDepth           = 16                           // audio wav encoder/decoder sample size
	ncolors            = 5                            // number of grayscale colors in spectrogram
	nffts              = 64                           // number of ffts in the spectrograms
	avgDuration        = 200                          // average duration in samples of the speech frame size
	npatterns          = 16                           // number of synthetic speech patterns to classify
	maxSubFreq         = 5                            // max number of sub-frequencies
)

// test statistics that are tabulated in HTML
type Results struct {
	Class   string // int
	Correct string // int      percent correct
	Count   string // int      number of training examples in the class
}

// Type to contain all the HTML template actions
type PlotT struct {
	Grid          []string  // plotting grid
	Status        string    // status of the plot
	Xlabel        []string  // x-axis labels
	Ylabel        []string  // y-axis labels
	Trials        string    // number of samples
	FFTSize       string    // 8192, 4098, 2048, 1024
	FFTWindow     string    // Bartlett, Welch, Hamming, Hanning, Rectangle
	Domain        string    // plot time or spectrogra domain
	TestResults   []Results // tabulated statistics of testing
	TotalCount    string    // Results tabulation
	TotalCorrect  string
	DelDuration   string // delta of the speech frame duration
	DelPitch      string // delta of the speech frame frequencies
	DelAmpl       string // delta of the speech frame frequency amplitudes
	PercentVoiced string // percentage of the speech frames voiced
	Patterns      string // number of speech patterns
	SpeechPattern string // speech pattern to display
}

// Type to hold the minimum and maximum data values
type Endpoints struct {
	xmin float64
	xmax float64
	ymin float64
	ymax float64
}

type Stats struct {
	correct    []int // % correct classifcation
	classCount []int // #samples in each class
}

// training examples
type Sample struct {
	desired int         // numerical class of the synthetic speech pattern
	data    [][]float64 //  frequency bins versus frame (time) from the STFT
}

// cross correlation result
type Result struct {
	xcorr   float64
	pattern int
}

// Speech frame attributes
type SpeechFrame struct {
	freqs []float64
	amps  []float64
}

// Primary data structure for holding the SPCC state
type SPCC struct {
	plot          *PlotT                 // data to be distributed in the HTML template
	Endpoints                            // embedded struct
	nsamples      int                    // number of synthetic speech pattern
	trials        int                    // number of samples
	domain        string                 // time or spectrogram plot
	data          []float64              // cross-entropy Loss in output layer per epoch used in Learning Curve
	words         []string               // classified words in test message
	grayscale     map[int]string         // grayscale for spectrogram
	fftSize       int                    // FFT size for spectrogram
	fftWindow     string                 // FFT window
	speechPat     [][]SpeechFrame        // speech pattern containing frequencies and amplitudes
	specRef       [npatterns][][]float64 // spectrogram references for cross correlation
	delPitch      int                    // pitch delta
	delDuration   int                    // duration delta in samples
	delAmpl       float64                // amplitude delta of the frequencies
	percentVoiced int                    // percent voiced
	synSpeech     []float64              // synthetic speech
	statistics    Stats
	freqs         []float64 // speech frame frequencies
	amps          []float64 // speech frame amplitudes
}

// Window function type
type Window func(n int, m int) complex128

// global variables for parse and execution of the html template
var (
	tmplTestingSPCC *template.Template
	tmplDisplaySPCC *template.Template
	winType         = []string{"Bartlett", "Welch", "Hamming", "Hanning", "Rectangle"}
)

// init parses the html template files
func init() {
	tmplTestingSPCC = template.Must(template.ParseFiles(fileTestingSPCC))
	tmplDisplaySPCC = template.Must(template.ParseFiles(fileDisplaySPCC))
}

// Bartlett window
func bartlett(n int, m int) complex128 {
	real := 1.0 - math.Abs((float64(n)-float64(m))/float64(m))
	return complex(real, 0)
}

// Welch window
func welch(n int, m int) complex128 {
	x := math.Abs((float64(n) - float64(m)) / float64(m))
	real := 1.0 - x*x
	return complex(real, 0)
}

// Hamming window
func hamming(n int, m int) complex128 {
	return complex(.54-.46*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Hanning window
func hanning(n int, m int) complex128 {
	return complex(.5-.5*math.Cos(math.Pi*float64(n)/float64(m)), 0)
}

// Rectangle window
func rectangle(n int, m int) complex128 {
	return 1.0
}

// xcorr performs correlation between the sample and the pattern speech reference
func (spcc *SPCC) xcorr(pattern int, sample *Sample, resultCollector chan<- Result) error {
	xcorr := 0.0
	// loop over the frames
	for frame := range spcc.specRef[pattern] {
		// loop over the frequency bins
		for bin := range spcc.specRef[pattern][frame] {
			xcorr += spcc.specRef[pattern][frame][bin] * sample.data[frame][bin]
		}
	}
	// send the result back to collector
	resultCollector <- Result{xcorr: xcorr, pattern: pattern}
	return nil
}

// classify the Sample using cross correlation with spectrogram references
func (spcc *SPCC) classifySample(sample *Sample) error {
	// buffered channel to collect results of cross correlation
	resultCollector := make(chan Result, npatterns)
	// loop over the spectrogram references
	for pattern := range npatterns {
		// launch goroutines to cross correlate against the spectrogram references
		go spcc.xcorr(pattern, sample, resultCollector)
	}

	maxXcorr := -math.MaxFloat64
	maxPattern := 0
	// collect results consisting of xcorr and pattern number and tabulate correct count
	for range npatterns {
		result := <-resultCollector
		if result.xcorr > maxXcorr {
			maxXcorr = result.xcorr
			maxPattern = result.pattern
		}
	}
	close(resultCollector)

	// Assign Stats.correct, Stats.classCount
	spcc.statistics.classCount[sample.desired]++
	if maxPattern == sample.desired {
		spcc.statistics.correct[maxPattern]++
	}

	return nil
}

// createSpectrogram creates spectrograms from the synthetic speech used in runTestingTrials
func (spcc *SPCC) createSpectrogram(samp *Sample) error {

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	PSD := make([]float64, spcc.fftSize/2)

	// Create the spectrogram, no overlap
	// loop over the samples, with fftSize jump
	for smpl := 0; smpl < len(spcc.synSpeech); smpl += spcc.fftSize {
		frame := smpl / spcc.fftSize
		_, psdMax, err := spcc.calculatePSD(spcc.synSpeech[smpl:smpl+spcc.fftSize], PSD, spcc.fftWindow, spcc.fftSize)
		if err != nil {
			fmt.Printf("calculatePSD error: %v\n", err)
			return fmt.Errorf("calculatePSD error: %v", err.Error())
		}
		val := 0.0
		// loop over the frequency bins, only want 0 < freq (Hz) < 1000
		// this corresponds to 0 < bin < 32
		for bin := 0; bin < spcc.fftSize/8; bin++ {
			// Relative power in the bin determines the grayscale color
			r := PSD[bin] / psdMax
			// five-color grayscale, 0 is white, 8 is black, convert to (0,8)
			if r < .1 {
				val = 0.0
			} else if r < .25 {
				val = 2.0
			} else if r < .50 {
				val = 4.0
			} else if r < .80 {
				val = 6.0
			} else {
				val = 8.0
			}
			samp.data[frame][bin] = val
		}
	}
	return nil
}

// create synthetic speech patterns consisting of 25ms frames of voiced or unvoiced input
func (spcc *SPCC) createPatterns() error {
	nsamples := spcc.fftSize * nffts
	// a block consists of avgDuration samples = 25ms of speech sampled at 8,000 Hz
	nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))
	const (
		amplMinf1  = 500.0
		amplMaxf1  = 1000.0
		f1Min      = 200   // Hz = cycles/sec
		f1Max      = 800   // Hz = cycles/sec
		sigmaNoise = 200.0 // unvoiced speech
		nyquist    = sampleRate / 2
		frameScale = 2 // frame extender
	)

	nsubfreq := 0

	// make SpeechFrames for the speech patterns
	spcc.speechPat = make([][]SpeechFrame, npatterns)
	for i := range spcc.speechPat {
		spcc.speechPat[i] = make([]SpeechFrame, nframes)
	}

	// Create the speech patterns for voiced or unvoiced frames
	// loop over number of speech patterns
	for pat := 0; pat < npatterns; pat++ {
		speechfile := filepath.Join(dataDir, fmt.Sprintf("speech%d.csv", pat))
		fspeech, err := os.Create(speechfile)
		if err != nil {
			return fmt.Errorf("createPatterns could not create file %s error: %s", speechfile, err.Error())
		}
		// loop over the number of frames
		for fr := 0; fr < nframes; fr++ {
			// select frequencies and amplitudes depending on voiced/unvoiced
			if fr%frameScale == 0 {
				if 100.0*rand.Float64() < float64(spcc.percentVoiced) {
					// voiced, cycles/sec = Hz
					// 1-5 subfrequencies
					nsubfreq = rand.Intn(maxSubFreq) + 1
					spcc.speechPat[pat][fr].freqs = make([]float64, nsubfreq+1)
					spcc.speechPat[pat][fr].amps = make([]float64, nsubfreq+1)
					// fundamental frequency
					spcc.speechPat[pat][fr].freqs[0] = f1Min + (f1Max-f1Min)*rand.Float64()
					spcc.speechPat[pat][fr].amps[0] = amplMinf1 + (amplMaxf1-amplMinf1)*rand.Float64()
					// subfrequencies
					for sf := 0; sf < nsubfreq; sf++ {
						spcc.speechPat[pat][fr].freqs[sf+1] =
							spcc.speechPat[pat][fr].freqs[sf] + (nyquist-spcc.speechPat[pat][fr].freqs[sf])*rand.Float64()
						spcc.speechPat[pat][fr].amps[sf+1] = spcc.speechPat[pat][fr].amps[sf] * 0.9
					}
				} else {
					// unvoiced is gaussian noise
					nsubfreq = 0
					spcc.speechPat[pat][fr].freqs = make([]float64, nsubfreq+1)
					spcc.speechPat[pat][fr].amps = make([]float64, nsubfreq+1)
					spcc.speechPat[pat][fr].freqs[0] = 0.0
					spcc.speechPat[pat][fr].amps[0] = sigmaNoise * rand.Float64()
				}
			} else {
				// Use previous frame parameters so that the frame is extended
				spcc.speechPat[pat][fr].freqs = make([]float64, nsubfreq+1)
				spcc.speechPat[pat][fr].amps = make([]float64, nsubfreq+1)
				// fundamental frequency
				spcc.speechPat[pat][fr].freqs[0] = spcc.speechPat[pat][fr-1].freqs[0]
				spcc.speechPat[pat][fr].amps[0] = spcc.speechPat[pat][fr-1].amps[0]
				// subfrequencies
				for sf := 0; sf < nsubfreq; sf++ {
					spcc.speechPat[pat][fr].freqs[sf+1] = spcc.speechPat[pat][fr-1].freqs[sf+1]
					spcc.speechPat[pat][fr].amps[sf+1] = spcc.speechPat[pat][fr-1].amps[sf+1]
				}
			}
			// Save the speech frame to disk file
			// frequencies
			for _, freq := range spcc.speechPat[pat][fr].freqs {
				_, err = fmt.Fprintf(fspeech, "%.16f,", freq)
				if err != nil {
					return fmt.Errorf("createPatterns file: %s, frame: %d, freq: %f write error: %s", speechfile, fr, freq, err.Error())
				}
			}
			// amplitudes
			for _, amp := range spcc.speechPat[pat][fr].amps[0 : len(spcc.speechPat[pat][fr].amps)-1] {
				_, err = fmt.Fprintf(fspeech, "%.16f,", amp)
				if err != nil {
					return fmt.Errorf("createPatterns file: %s, frame: %d, ampl: %f write error: %s", speechfile, fr, amp, err.Error())
				}
			}
			_, err = fmt.Fprintf(fspeech, "%.16f\n", spcc.speechPat[pat][fr].amps[len(spcc.speechPat[pat][fr].amps)-1])
			if err != nil {
				return fmt.Errorf("createPatterns file: %s, frame: %d write error: %s", speechfile, fr, err.Error())
			}
		}
		fspeech.Close()
	}
	return nil
}

// synthesize creates synthetic speech using frequencies and amplitudes of sinusoids or gaussian noise
func (spcc *SPCC) synthesize(nfreqs int, start int, stop int) error {

	t := 0.0
	step := 1.0 / float64(sampleRate)
	var sum float64
	// calculate speech over the interval
	if start == 0 {
		sum = 0.0
		if nfreqs == 1 {
			sum += spcc.amps[0] * rand.NormFloat64()
		} else {
			for j := 0; j < nfreqs; j++ {
				sum += spcc.amps[j] * math.Sin(2.0*math.Pi*spcc.freqs[j]*t)
			}
		}
		t += step
		spcc.synSpeech[0] = sum
		start++
	}
	for i := start; i < stop; i++ {
		sum = 0.0
		if nfreqs == 1 {
			sum += spcc.amps[0] * rand.NormFloat64()
		} else {
			for j := 0; j < nfreqs; j++ {
				sum += spcc.amps[j] * math.Sin(2.0*math.Pi*spcc.freqs[j]*t)
			}
		}
		t += step
		spcc.synSpeech[i] = 0.5 * (sum + spcc.synSpeech[i-1])
	}
	return nil
}

// createSpeech creates a slice of training/testing synthetic speech based on the speech patterns
func (spcc *SPCC) createSpeech(pattern int) error {
	// use the speech patterns and apply deltas for the frequencies, amplitude, and duration so the SPCC generalizes
	nsamples := spcc.fftSize * nffts
	remain := nsamples
	nframes := len(spcc.speechPat[pattern])
	// starting sample for current frame
	samp := 0

	const (
		margin1       int = avgDuration - 40
		margin2       int = 2 * margin1
		delSigmaNoise     = 1.0
	)

	// loop over the frame of the pattern and retrieve the attributes of the speech frame
	// add delta for pitch and duration
	for frame := 0; frame < nframes-1; frame++ {
		if remain < margin2 {
			return fmt.Errorf("repeat")
		}
		nfreqs := len(spcc.speechPat[pattern][frame].freqs)
		// unvoiced pattern, frequency = 0
		if nfreqs == 1 {
			if rand.Intn(2) > 0 {
				spcc.amps[0] = spcc.speechPat[pattern][frame].amps[0] + delSigmaNoise*spcc.delAmpl
			} else {
				spcc.amps[0] = spcc.speechPat[pattern][frame].amps[0] - delSigmaNoise*spcc.delAmpl
			}
			spcc.freqs[0] = spcc.speechPat[pattern][frame].freqs[0]
		} else {
			// voiced pattern
			for i := 0; i < nfreqs; i++ {
				if rand.Intn(2) > 0 {
					spcc.freqs[i] =
						spcc.speechPat[pattern][frame].freqs[i] + float64(spcc.delPitch)
				} else {
					spcc.freqs[i] =
						spcc.speechPat[pattern][frame].freqs[i] - float64(spcc.delPitch)
				}

				if rand.Intn(2) > 0 {
					spcc.amps[i] = spcc.speechPat[pattern][frame].amps[i] * (1.0 + float64(spcc.delAmpl))
				} else {
					spcc.amps[i] = spcc.speechPat[pattern][frame].amps[i] * (1.0 - float64(spcc.delAmpl))
				}
			}
		}
		duration := avgDuration
		// add or subtract delta
		if rand.Intn(2) > 0 {
			duration += spcc.delDuration
		} else {
			duration -= spcc.delDuration
		}
		remain -= duration

		err := spcc.synthesize(nfreqs, samp, samp+duration)
		if err != nil {
			return fmt.Errorf("synthesize error: %v", err.Error())
		}
		samp += duration
	}

	if remain < margin1 {
		return fmt.Errorf("repeat")
	}

	nfreqs := len(spcc.speechPat[pattern][nframes-1].freqs)
	// unvoiced pattern, frequency = 0
	if nfreqs == 1 {
		if rand.Intn(2) > 0 {
			spcc.amps[0] = spcc.speechPat[pattern][nframes-1].amps[0] + delSigmaNoise*spcc.delAmpl
		} else {
			spcc.amps[0] = spcc.speechPat[pattern][nframes-1].amps[0] - delSigmaNoise*spcc.delAmpl
		}
		spcc.freqs[0] = spcc.speechPat[pattern][nframes-1].freqs[0]
	} else {
		// voiced pattern
		for i := 1; i < nfreqs; i++ {
			if rand.Intn(2) > 0 {
				spcc.freqs[i] =
					spcc.speechPat[pattern][nframes-1].freqs[i] + float64(spcc.delPitch)/spcc.speechPat[pattern][nframes-1].freqs[i]
			} else {
				spcc.freqs[i] =
					spcc.speechPat[pattern][nframes-1].freqs[i] - float64(spcc.delPitch)/spcc.speechPat[pattern][nframes-1].freqs[i]
			}

			if rand.Intn(2) > 0 {
				spcc.amps[i] = spcc.speechPat[pattern][nframes-1].amps[i] * (1.0 + float64(spcc.delAmpl))
			} else {
				spcc.amps[i] = spcc.speechPat[pattern][nframes-1].amps[i] * (1.0 - float64(spcc.delAmpl))
			}
		}
	}

	err := spcc.synthesize(nfreqs, samp, samp+remain)
	if err != nil {
		return fmt.Errorf("synthesize error: %v", err.Error())
	}

	return nil
}

// newSPCC constructs an SPCC instance for testing
func newSPCC(r *http.Request, trials int, plot *PlotT) (*SPCC, error) {
	// Read the testing parameters in the HTML Form

	fftWindow := r.FormValue("fftwindow")

	txt := r.FormValue("fftsize")
	fftSize, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("fftsize int conversion error: %v\n", err)
		return nil, err
	}

	// Get delta pitch, delta duration, delta amplitude, and percent voiced speech
	txt = r.FormValue("delpitch")
	delPitch, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("delta Pitch conversion error: %v\n", err)
		return nil, err
	}

	txt = r.FormValue("delduration")
	delDuration, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("delta Duration conversion error: %v\n", err)
		return nil, err
	}

	txt = r.FormValue("delampl")
	delAmpl, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("delta Amplitude conversion error: %v\n", err)
		return nil, err
	}

	txt = r.FormValue("percentvoiced")
	percentVoiced, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("percent Voiced conversion error: %v\n", err)
		return nil, err
	}

	spcc := SPCC{
		trials: trials,
		plot:   plot,
		Endpoints: Endpoints{
			ymin: 0.0,
			ymax: 100.0,
			xmin: 0,
			xmax: float64(npatterns - 1)},
		words:         make([]string, 0),
		fftSize:       fftSize,
		fftWindow:     fftWindow,
		delDuration:   delDuration,
		delPitch:      delPitch,
		delAmpl:       delAmpl,
		percentVoiced: percentVoiced,
		freqs:         make([]float64, maxSubFreq+1),
		amps:          make([]float64, maxSubFreq+1),
	}

	// number of samples per pattern, number of frames in the pattern
	nsamples := spcc.fftSize * nffts
	nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))
	// number of PSD bins to cross correlate, dimension reduction
	nbins := spcc.fftSize / 8

	// synthetic speech for creating speech with synthesize
	spcc.synSpeech = make([]float64, nsamples)

	// spectrogram references for cross correlation
	for pattern := range spcc.specRef {
		spcc.specRef[pattern] = make([][]float64, nframes)
		for frame := range spcc.specRef[pattern] {
			spcc.specRef[pattern][frame] = make([]float64, nbins)
		}
	}

	// percent correct classification of speech patterns
	spcc.data = make([]float64, npatterns)

	return &spcc, nil
}

// gridFillInterp inserts the data points in the grid and draws a straight line between points
func (spcc *SPCC) gridFillInterp() error {
	var (
		x            float64 = 0.0
		y            float64 = spcc.data[0]
		prevX, prevY float64
		xscale       float64
		yscale       float64
	)

	// Mark the data x-y coordinate online at the corresponding
	// grid row/column.

	// Calculate scale factors for x and y
	xscale = float64(cols-1) / (spcc.xmax - spcc.xmin)
	yscale = float64(rows-1) / (spcc.ymax - spcc.ymin)

	spcc.plot.Grid = make([]string, rows*cols)

	// This cell location (row,col) is on the line
	row := int((spcc.ymax-y)*yscale + .5)
	col := int((x-spcc.xmin)*xscale + .5)
	spcc.plot.Grid[row*cols+col] = "online"

	prevX = x
	prevY = y

	// Scale factor to determine the number of interpolation points
	lenEPy := spcc.ymax - spcc.ymin
	lenEPx := spcc.xmax - spcc.xmin

	// Continue with the rest of the points in the file
	for i := 1; i < len(spcc.data); i++ {
		x++
		// mse/epoch or percent-correct/pattern
		y = spcc.data[i]

		// This cell location (row,col) is on the line
		row := int((spcc.ymax-y)*yscale + .5)
		col := int((x-spcc.xmin)*xscale + .5)
		spcc.plot.Grid[row*cols+col] = "online"

		// Interpolate the points between previous point and current point

		/* lenEdge := math.Sqrt((x-prevX)*(x-prevX) + (y-prevY)*(y-prevY)) */
		lenEdgeX := math.Abs((x - prevX))
		lenEdgeY := math.Abs(y - prevY)
		ncellsX := int(float64(cols) * lenEdgeX / lenEPx) // number of points to interpolate in x-dim
		ncellsY := int(float64(rows) * lenEdgeY / lenEPy) // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := max(ncellsY, ncellsX)

		stepX := (x - prevX) / float64(ncells)
		stepY := (y - prevY) / float64(ncells)

		// loop to draw the points
		interpX := prevX
		interpY := prevY
		for i := 0; i < ncells; i++ {
			row := int((spcc.ymax-interpY)*yscale + .5)
			col := int((interpX-spcc.xmin)*xscale + .5)
			spcc.plot.Grid[row*cols+col] = "online"
			interpX += stepX
			interpY += stepY
		}

		// Update the previous point with the current point
		prevX = x
		prevY = y
	}
	return nil
}

// insertLabels inserts x- an y-axis labels in the plot
func (spcc *SPCC) insertLabels() {
	spcc.plot.Xlabel = make([]string, xlabels)
	spcc.plot.Ylabel = make([]string, ylabels)
	// Construct x-axis labels
	incr := (spcc.xmax - spcc.xmin) / (xlabels - 1)
	x := spcc.xmin
	// First label is empty for alignment purposes
	for i := range spcc.plot.Xlabel {
		spcc.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (spcc.ymax - spcc.ymin) / (ylabels - 1)
	y := spcc.ymin
	for i := range spcc.plot.Ylabel {
		spcc.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}
}

// createSpectRef creates spectrogram references for xcorr
func (spcc *SPCC) createSpectRef() error {
	// save the parameter deltas to be restored
	delAmpl := spcc.delAmpl
	delDuration := spcc.delDuration
	delPitch := spcc.delPitch
	// set the deltas to zero
	spcc.delAmpl = 0.0
	spcc.delDuration = 0.0
	spcc.delPitch = 0.0
	// number of samples per pattern, number of frames in the pattern
	nsamples := spcc.fftSize * nffts
	nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))
	// number of PSD bins to cross correlate
	nbins := spcc.fftSize / 8

	// Create a sample for containing the spectrogram for
	// classification using cross correlation
	samp := Sample{data: make([][]float64, nframes)}
	for i := range samp.data {
		samp.data[i] = make([]float64, nbins)
	}

	for pattern := range npatterns {
		err := spcc.createSpeech(pattern)
		if err != nil {
			return fmt.Errorf("createSpeech error: %v", err.Error())
		}
		samp.desired = pattern
		err = spcc.createSpectrogram(&samp)
		if err != nil {
			return fmt.Errorf("createSpectrogram error: %v", err.Error())
		}

		// Copy the sample to spcc
		for frame := range samp.data {
			copy(spcc.specRef[pattern][frame], samp.data[frame])
		}
	}

	// restore deltas
	spcc.delAmpl = delAmpl
	spcc.delDuration = delDuration
	spcc.delPitch = delPitch

	return nil
}

// handleSPCC performs cross correlation of the sample spectrogram against pattern references
func handleSPCC(w http.ResponseWriter, r *http.Request) {
	// create synthetic speech
	// loop over the synthetic speech and generate the spectrogram
	// propagate forward and classify the output
	// fill the grid with the percent correct

	var (
		plot PlotT
		spcc *SPCC
	)

	// Get the number of trials
	txt := r.FormValue("trials")
	// Need trials to continue
	if len(txt) > 0 {
		trials, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("Trials int conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Trials conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTestingSPCC.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// create SPCC instance to hold state
		spcc, err = newSPCC(r, trials, &plot)
		if err != nil {
			fmt.Printf("newSPCC() error: %v\n", err)
			plot.Status = fmt.Sprintf("newSPCC() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTestingSPCC.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// create new synthetic speech patterns
		newPattern := r.FormValue("speechpattern")
		if newPattern == "new" {
			if err = spcc.createPatterns(); err != nil {
				fmt.Printf("createPatterns() error: %v\n", err)
				plot.Status = fmt.Sprintf("createPatterns() error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplTestingSPCC.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			// Read the synthetic speech patterns
		} else {
			files, err := os.ReadDir(dataDir)
			if err != nil {
				fmt.Printf("ReadDir %s error: %v\n", dataDir, err)
				plot.Status = fmt.Sprintf("ReadDir %s error: %v", dataDir, err.Error())
				// Write to HTTP using template and grid
				if err := tmplTestingSPCC.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			if len(files) == 0 {
				fmt.Printf("No synthetic speech files in %s\n", dataDir)
				plot.Status = fmt.Sprintf("No filter files in %s", dataDir)
				// Write to HTTP using template and grid
				if err := tmplTestingSPCC.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			} else {
				// make SpeechFrames for the speech patterns
				nsamples := spcc.fftSize * nffts
				// a frame consists of avgDuration samples = 25ms of speech sampled at 8,000 Hz
				nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))
				spcc.speechPat = make([][]SpeechFrame, npatterns)
				for i := range spcc.speechPat {
					spcc.speechPat[i] = make([]SpeechFrame, nframes)
				}
				pattern := 0
				// Retrieve the speech files
				for _, dirEntry := range files {
					name := dirEntry.Name()
					if strings.Contains(name, "speech") && strings.Contains(name, "csv") {
						fspeech, err := os.Open(filepath.Join(dataDir, name))
						if err != nil {
							fmt.Printf("Open %s error: %v\n", name, err)
							plot.Status = fmt.Sprintf("Open %s error: %v", name, err.Error())
							// Write to HTTP using template and grid
							if err := tmplTestingSPCC.Execute(w, plot); err != nil {
								log.Fatalf("Write to HTTP output using template with error: %v\n", err)
							}
							return
						}
						scanner := bufio.NewScanner(fspeech)
						frame := 0
						for scanner.Scan() {
							line := scanner.Text()
							items := strings.Split(line, ",")
							nfreqs := len(items) / 2
							spcc.speechPat[pattern][frame].freqs = make([]float64, nfreqs)
							spcc.speechPat[pattern][frame].amps = make([]float64, nfreqs)
							for i := 0; i < nfreqs; i++ {
								freq, err := strconv.ParseFloat(items[i], 64)
								if err != nil {
									plot.Status = fmt.Sprintf("freq %d conversion error: %v", i, err.Error())
									// Write to HTTP using template and grid
									if err := tmplTestingSPCC.Execute(w, plot); err != nil {
										log.Fatalf("Write to HTTP output using template with error: %v\n", err)
									}
									return
								}
								ampl, err := strconv.ParseFloat(items[i+nfreqs], 64)
								if err != nil {
									plot.Status = fmt.Sprintf("ampl %d conversion error: %v", i, err.Error())
									// Write to HTTP using template and grid
									if err := tmplTestingSPCC.Execute(w, plot); err != nil {
										log.Fatalf("Write to HTTP output using template with error: %v\n", err)
									}
									return
								}
								spcc.speechPat[pattern][frame].amps[i] = ampl
								spcc.speechPat[pattern][frame].freqs[i] = freq
							}
							frame++
						}
						fspeech.Close()
						if err = scanner.Err(); err != nil {
							fmt.Printf("speech file scanner error: %s", err.Error())
							// Write to HTTP using template and grid
							if err := tmplTestingSPCC.Execute(w, plot); err != nil {
								log.Fatalf("Write to HTTP output using template with error: %v\n", err)
							}
							return
						}
						pattern++
					}
				}
			}
		}

		// create the spectrogram references for xcorr
		err = spcc.createSpectRef()
		if err != nil {
			fmt.Printf("createSpectRef() error: %v\n", err)
			plot.Status = fmt.Sprintf("createSpectRef() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTestingSPCC.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Loop over the Trials
		err = spcc.runTestingTrials()
		if err != nil {
			fmt.Printf("runTestingTrials() error: %v\n", err)
			plot.Status = fmt.Sprintf("runTestingTrials() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTestingSPCC.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Put the percent correct in data
		for pat := range spcc.data {
			spcc.data[pat] = float64(spcc.statistics.correct[pat]) / float64(spcc.statistics.classCount[pat]) * 100.0
		}
		// Put data in PlotT
		err = spcc.gridFillInterp()
		if err != nil {
			fmt.Printf("gridFillInterp() error: %v\n", err)
			plot.Status = fmt.Sprintf("gridFillInterp() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTestingSPCC.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// insert x-labels and y-labels in PlotT
		spcc.insertLabels()

		// At the end of all trials, insert form previous control items in PlotT
		spcc.plot.Trials = strconv.Itoa(spcc.trials)
		spcc.plot.DelDuration = strconv.Itoa(spcc.delDuration)
		spcc.plot.DelPitch = strconv.Itoa(spcc.delPitch)
		spcc.plot.PercentVoiced = strconv.Itoa(spcc.percentVoiced)
		spcc.plot.DelAmpl = strconv.FormatFloat(spcc.delAmpl, 'f', 3, 64)

		spcc.plot.Status = "Testing results completed."

		// Execute data on HTML template
		if err = tmplTestingSPCC.Execute(w, spcc.plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	} else {
		plot.Status = "Enter Spectrogram Cross Correlation testing parameters."
		// Write to HTTP using template and grid
		if err := tmplTestingSPCC.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}
}

// Welch's Method and Bartlett's Method variation of the Periodogram
func (spcc *SPCC) calculatePSD(audio []float64, PSD []float64, fftWindow string, fftSize int) (int, float64, error) {

	N := fftSize
	m := N / 2

	// map of window functions
	window := make(map[string]Window, len(winType))
	// Put the window functions in the map
	window["Bartlett"] = bartlett
	window["Welch"] = welch
	window["Hamming"] = hamming
	window["Hanning"] = hanning
	window["Rectangle"] = rectangle

	w, ok := window[fftWindow]
	if !ok {
		fmt.Printf("Invalid FFT window type: %v\n", fftWindow)
		return 0, 0, fmt.Errorf("invalid FFT window type: %v", fftWindow)
	}

	bufN := make([]complex128, N)

	for j := 0; j < len(audio); j++ {
		bufN[j] = complex(audio[j], 0)
	}

	// zero-pad the remaining samples
	for i := len(audio); i < N; i++ {
		bufN[i] = 0
	}

	// window the N samples with chosen window
	for k := 0; k < N; k++ {
		bufN[k] *= w(k, m)
	}

	// Perform N-point complex FFT and add squares to previous values in PSD
	fourierN := fft.FFT(bufN)
	x := cmplx.Abs(fourierN[0])
	PSD[0] = x * x
	psdMax := PSD[0]
	binMax := 0
	for j := 1; j < m; j++ {
		// Use positive and negative frequencies -> bufN[N-j] = bufN[-j]
		xj := cmplx.Abs(fourierN[j])
		xNj := cmplx.Abs(fourierN[N-j])
		PSD[j] = xj*xj + xNj*xNj
		if PSD[j] > psdMax {
			psdMax = PSD[j]
			binMax = j
		}
	}

	return binMax, psdMax, nil
}

// runTestingTrials classifies test examples and tabulates test results
func (spcc *SPCC) runTestingTrials() error {

	// number of samples per pattern, number of frames in the pattern
	nsamples := spcc.fftSize * nffts
	nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))
	// number of PSD bins to cross correlate
	nbins := spcc.fftSize / 8

	// Create a sample for containing the spectrogram for
	// classification using cross correlation
	samp := Sample{data: make([][]float64, nframes)}
	for i := range samp.data {
		samp.data[i] = make([]float64, nbins)
	}
	spcc.statistics =
		Stats{correct: make([]int, npatterns), classCount: make([]int, npatterns)}

	for n := 0; n < spcc.trials; n++ {
		// create speech for one pattern
		// Randomly choose a pattern
		pattern := rand.Intn(npatterns)
		err := spcc.createSpeech(pattern)
		for err != nil {
			if err.Error() == "repeat" {
				err = spcc.createSpeech(pattern)
			} else {
				fmt.Printf("createSpeech error: %v\n", err.Error())
				return fmt.Errorf("createSpeech error: %v", err.Error())
			}
		}

		samp.desired = pattern

		// create spectrogram for the speech sample
		err = spcc.createSpectrogram(&samp)
		if err != nil {
			fmt.Printf("createSpectrogram error: %v\n", err.Error())
			return fmt.Errorf("createSpectrogram error: %v", err.Error())
		}

		// classify the speech sample
		err = spcc.classifySample(&samp)
		if err != nil {
			fmt.Printf("classifySample error: %v\n", err.Error())
			return fmt.Errorf("classifySample error: %v", err.Error())
		}
	}

	spcc.plot.TestResults = make([]Results, npatterns)

	totalCount := 0
	totalCorrect := 0
	classCount := 0
	// tabulate TestResults by converting numbers to string in Results
	for i := range spcc.plot.TestResults {
		classCount = spcc.statistics.classCount[i]
		totalCount += classCount
		totalCorrect += spcc.statistics.correct[i]
		if classCount > 0 {
			spcc.plot.TestResults[i] = Results{
				Class:   strconv.Itoa(i),
				Count:   strconv.Itoa(classCount),
				Correct: strconv.Itoa(spcc.statistics.correct[i] * 100 / classCount),
			}
		} else {
			spcc.plot.TestResults[i] = Results{
				Class:   strconv.Itoa(i),
				Count:   strconv.Itoa(classCount),
				Correct: "0",
			}
		}
	}
	spcc.plot.TotalCount = strconv.Itoa(totalCount)
	spcc.plot.TotalCorrect = strconv.Itoa(totalCorrect * 100 / totalCount)
	spcc.plot.Trials = strconv.Itoa(spcc.trials)

	spcc.plot.Status = "Testing results completed."

	spcc.plot.Trials = strconv.Itoa(spcc.trials)
	spcc.plot.FFTSize = strconv.Itoa(spcc.fftSize)
	spcc.plot.FFTWindow = spcc.fftWindow
	spcc.plot.Patterns = strconv.Itoa(npatterns)
	spcc.plot.DelPitch = strconv.Itoa(spcc.delPitch)
	spcc.plot.DelDuration = strconv.Itoa(spcc.delDuration)
	spcc.plot.PercentVoiced = strconv.Itoa(spcc.percentVoiced)
	spcc.plot.DelAmpl = strconv.FormatFloat(spcc.delAmpl, 'f', -1, 64)

	return nil
}

// findEndpoints finds the minimum and maximum data values
func (ep *Endpoints) findEndpoints(input []float64) {
	ep.ymax = -math.MaxFloat64
	ep.ymin = math.MaxFloat64
	for _, y := range input {

		if y > ep.ymax {
			ep.ymax = y
		}
		if y < ep.ymin {
			ep.ymin = y
		}
	}
}

// processTimeDomain plots the time domain data of the synthetic speech
func (spcc *SPCC) processTimeDomain(pattern int) error {

	var (
		xscale    float64
		yscale    float64
		endpoints Endpoints
	)

	spcc.plot.Grid = make([]string, rows*cols)
	spcc.plot.Xlabel = make([]string, xlabels)
	spcc.plot.Ylabel = make([]string, ylabels)

	spcc.nsamples = len(spcc.synSpeech)

	endpoints.findEndpoints(spcc.synSpeech)
	// time starts at 0 and ends at #samples*sampling period
	endpoints.xmin = 0.0
	// #samples*sampling period, sampling period = 1/sampleRate
	endpoints.xmax = float64(spcc.nsamples) / float64(sampleRate)

	// EP means endpoints
	lenEPx := endpoints.xmax - endpoints.xmin
	lenEPy := endpoints.ymax - endpoints.ymin
	prevTime := 0.0
	prevAmpl := spcc.synSpeech[0]

	// Calculate scale factors for x and y
	xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
	yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

	// This previous cell location (row,col) is on the line (visible)
	row := int((endpoints.ymax-spcc.synSpeech[0])*yscale + .5)
	col := int((0.0-endpoints.xmin)*xscale + .5)
	spcc.plot.Grid[row*cols+col] = "online"

	// Store the amplitude in the plot Grid
	for n := 1; n < spcc.nsamples; n++ {
		// Current time
		currTime := float64(n) / float64(sampleRate)

		// This current cell location (row,col) is on the line (visible)
		row := int((endpoints.ymax-spcc.synSpeech[n])*yscale + .5)
		col := int((currTime-endpoints.xmin)*xscale + .5)
		spcc.plot.Grid[row*cols+col] = "online"

		// Interpolate the points between previous point and current point;
		// draw a straight line between points.
		lenEdgeTime := math.Abs((currTime - prevTime))
		lenEdgeAmpl := math.Abs(spcc.synSpeech[n] - prevAmpl)
		ncellsTime := int(float64(cols) * lenEdgeTime / lenEPx) // number of points to interpolate in x-dim
		ncellsAmpl := int(float64(rows) * lenEdgeAmpl / lenEPy) // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := max(ncellsAmpl, ncellsTime)

		stepTime := float64(currTime-prevTime) / float64(ncells)
		stepAmpl := float64(spcc.synSpeech[n]-prevAmpl) / float64(ncells)

		// loop to draw the points
		interpTime := prevTime
		interpAmpl := prevAmpl
		for i := 0; i < ncells; i++ {
			row := int((endpoints.ymax-interpAmpl)*yscale + .5)
			col := int((interpTime-endpoints.xmin)*xscale + .5)
			// This cell location (row,col) is on the line (visible)
			spcc.plot.Grid[row*cols+col] = "online"
			interpTime += stepTime
			interpAmpl += stepAmpl
		}

		// Update the previous point with the current point
		prevTime = currTime
		prevAmpl = spcc.synSpeech[n]

	}

	// Set plot status if no errors
	if len(spcc.plot.Status) == 0 {
		spcc.plot.Status = fmt.Sprintf("speech pattern %d plotted from (%.3f,%.3f) to (%.3f,%.3f)",
			pattern, endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
	}

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / (xlabels - 1)
	x := endpoints.xmin
	// First label is empty for alignment purposes
	for i := range spcc.plot.Xlabel {
		spcc.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	for i := range spcc.plot.Ylabel {
		spcc.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}

	return nil
}

// processSpectrogram creates a spectrogram of the speech waveform
func (spcc *SPCC) processSpectrogram(pattern int, fftWindow string, fftSize int) error {

	// get speech samples from spcc.synSpeech
	var (
		endpoints Endpoints
		PSD       []float64 // power spectral density
		xscale    float64   // data to grid in x direction
		yscale    float64   // data to grid in y direction
	)
	fftSize2 := fftSize / 2

	spcc.plot.Grid = make([]string, rows*cols)
	spcc.plot.Xlabel = make([]string, xlabels)
	spcc.plot.Ylabel = make([]string, ylabels)

	// Power Spectral Density, PSD[N/2] is the Nyquist critical frequency
	// It is (sampling frequency)/2, the highest non-aliased frequency
	PSD = make([]float64, fftSize/2)

	spcc.nsamples = len(spcc.synSpeech)
	// x-axis is time or sample, y-axis is frequency
	endpoints.xmin = 0.0
	endpoints.xmax = float64(spcc.nsamples)
	endpoints.ymin = 0.0
	endpoints.ymax = float64(fftSize2) // equivalent to Nyquist critical frequency

	// Calculate scale factors to convert physical units to screen units
	xscale = float64(cols-1) / (endpoints.xmax - endpoints.xmin)
	yscale = float64(rows-1) / (endpoints.ymax - endpoints.ymin)

	// number of cells to interpolate in time and frequency, stepping by fftSize/2 in time for ncellst
	// round up so the cells in the plot grid are connected
	ncellst := int((math.Ceil(float64(cols) * float64(fftSize2) / float64(spcc.nsamples))))
	ncellsf := int(math.Ceil(float64(rows) / float64((fftSize2))))

	stepTime := float64((fftSize) / ncellst)
	stepFreq := 1.0 / float64(ncellsf)

	// for loop over samples, increment by fftSize/2, calculatePSD on the batch
	// Overlap by 50% due to non-rectangular window to avoid Gibbs phenomenon
	for smpl := 0; smpl < spcc.nsamples; smpl += fftSize2 {
		// calculate the PSD using Bartlett's or Welch's variant of the Periodogram
		end := smpl + fftSize
		if end > spcc.nsamples {
			end = spcc.nsamples
		}
		_, psdMax, err := spcc.calculatePSD(spcc.synSpeech[smpl:end], PSD, fftWindow, fftSize)
		if err != nil {
			fmt.Printf("calculatePSD error: %v\n", err)
			return fmt.Errorf("calculatePSD error: %v", err.Error())
		}

		// for loop over the frequency bins in the PSD
		for bin := 0; bin < fftSize2; bin++ {
			// find the grayscale color based on bin power
			// largest power is black, smallest power is white
			// shades of gray in-between black and white
			var gs string
			r := PSD[bin] / psdMax
			if r < .10 {
				gs = spcc.grayscale[4]
			} else if r < .25 {
				gs = spcc.grayscale[3]
			} else if r < .50 {
				gs = spcc.grayscale[2]
			} else if r < .80 {
				gs = spcc.grayscale[1]
			} else {
				gs = spcc.grayscale[0]
			}

			// interpolate in time
			interpTime := float64(smpl)
			for nct := 0; nct < ncellst; nct++ {
				col := int((interpTime-endpoints.xmin)*xscale + .5)
				if col >= cols {
					col = cols - 1
				}
				// interpolate in frequency
				interpFreq := float64(bin)
				for ncf := 0; ncf < ncellsf; ncf++ {
					row := int((endpoints.ymax-interpFreq)*yscale + .5)
					if row < 0 {
						row = 0
					}
					// Store the color in the plot Grid
					spcc.plot.Grid[row*cols+col] = gs
					interpFreq += stepFreq
				}
				interpTime += stepTime
			}
		}
	}

	// Set plot status if no errors
	if len(spcc.plot.Status) == 0 {
		spcc.plot.Status = fmt.Sprintf("spectrogram of pattern %d plotted from (%.3f,%.3f) to (%.3f,%.3f)",
			pattern, endpoints.xmin, endpoints.ymin, endpoints.xmax, endpoints.ymax)
	}

	// Construct x-axis labels
	incr := (endpoints.xmax - endpoints.xmin) / ((xlabels - 1) * sampleRate)
	x := endpoints.xmin / sampleRate
	// First label is empty for alignment purposes
	for i := range spcc.plot.Xlabel {
		spcc.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Apply the  sampling rate in Hz to the y-axis using a scale factor
	// Convert the fft size to sampleRate/2, the Nyquist critical frequency
	sf := 0.5 * sampleRate / endpoints.ymax

	// Construct y-axis labels
	incr = (endpoints.ymax - endpoints.ymin) / (ylabels - 1)
	y := endpoints.ymin
	// First label is empty for alignment purposes
	for i := range spcc.plot.Ylabel {
		spcc.plot.Ylabel[i] = fmt.Sprintf("%.0f", y*sf)
		y += incr
	}

	return nil
}

// newDisplaySPCC creates a SPCC instance for displaying speech in time or spectrogram
func newDisplaySPCC(r *http.Request, plot *PlotT) (*SPCC, error) {

	// Get from form percent voiced, pitch variation, duration variation
	txt := r.FormValue("delpitch")
	delPitch, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("delPitch int conversion error: %v\n", err)
		return nil, fmt.Errorf("delPitch conversion to int error: %v", err.Error())
	}

	txt = r.FormValue("delduration")
	delDuration, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("delDuration int conversion error: %v\n", err)
		return nil, fmt.Errorf("delDuration conversion to int error: %v", err.Error())
	}

	txt = r.FormValue("percentvoiced")
	percentVoiced, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("percentVoiced int conversion error: %v\n", err)
		return nil, fmt.Errorf("percentVoiced conversion to int error: %v", err.Error())
	}

	txt = r.FormValue("delampl")
	delAmpl, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("delta Amplitude conversion error: %v\n", err)
		return nil, err
	}

	spcc := SPCC{
		plot:          plot,
		delPitch:      delPitch,
		delDuration:   delDuration,
		percentVoiced: percentVoiced,
		fftSize:       256,
		fftWindow:     "Rectangle",
		freqs:         make([]float64, maxSubFreq+1),
		amps:          make([]float64, maxSubFreq+1),
		delAmpl:       delAmpl,
	}

	// Determine if time or spectrogram domain plot
	spcc.domain = r.FormValue("domain")
	if spcc.domain == "spectrogram" {
		plot.Domain = "Spectrogram (Hz/sec)"
	} else {
		plot.Domain = "Time Domain (sec)"
	}

	// synthetic speech for creating speech with synthesize
	spcc.synSpeech = make([]float64, spcc.fftSize*nffts)

	return &spcc, nil
}

// handleDisplaySPCC displays the selected speech pattern (time or spectrogram) and plays the audio
func handleDisplaySPCC(w http.ResponseWriter, r *http.Request) {
	var (
		plot PlotT
		spcc *SPCC
	)

	// Get the speech pattern to display
	txt := r.FormValue("speechpattern")
	// Need speech pattern to continue
	if len(txt) > 0 {
		speechPattern, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("Speech Pattern int conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Speech Pattern conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTestingSPCC.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Construct SPCC instance containing SPCC state
		spcc, err = newDisplaySPCC(r, &plot)
		if err != nil {
			fmt.Printf("newDisplaySPCC() error: %v\n", err)
			plot.Status = fmt.Sprintf("newDisplaySPCC() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplDisplaySPCC.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// make SpeechFrames for the speech patterns
		nsamples := spcc.fftSize * nffts
		// a frame consists of avgDuration samples = 25ms of speech sampled at 8,000 Hz
		nframes := int(math.Ceil(float64(nsamples) / float64(avgDuration)))

		// create new synthetic speech patterns
		newPattern := r.FormValue("newpattern")
		if len(newPattern) > 0 {
			if err = spcc.createPatterns(); err != nil {
				fmt.Printf("createPatterns() error: %v\n", err)
				plot.Status = fmt.Sprintf("createPatterns() error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplaySPCC.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			// Read the synthetic speech patterns
		} else {
			files, err := os.ReadDir(dataDir)
			if err != nil {
				fmt.Printf("ReadDir %s error: %v\n", dataDir, err)
				plot.Status = fmt.Sprintf("ReadDir %s error: %v", dataDir, err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplaySPCC.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			if len(files) == 0 {
				fmt.Printf("No synthetic speech files in %s\n", dataDir)
				plot.Status = fmt.Sprintf("No synthetic speech files in %s, create new patterns", dataDir)
				// Write to HTTP using template and grid
				if err := tmplDisplaySPCC.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			} else {
				spcc.speechPat = make([][]SpeechFrame, npatterns)
				for i := range spcc.speechPat {
					spcc.speechPat[i] = make([]SpeechFrame, nframes)
				}
				pattern := 0
				// Retrieve the speech files
				for _, dirEntry := range files {
					name := dirEntry.Name()
					if strings.Contains(name, "speech") && strings.Contains(name, "csv") {
						fspeech, err := os.Open(filepath.Join(dataDir, name))
						if err != nil {
							fmt.Printf("Open %s error: %v\n", name, err)
							plot.Status = fmt.Sprintf("Open %s error: %v", name, err.Error())
							// Write to HTTP using template and grid
							if err := tmplTestingSPCC.Execute(w, plot); err != nil {
								log.Fatalf("Write to HTTP output using template with error: %v\n", err)
							}
							return
						}
						scanner := bufio.NewScanner(fspeech)
						frame := 0
						for scanner.Scan() {
							line := scanner.Text()
							items := strings.Split(line, ",")
							nfreqs := len(items) / 2
							spcc.speechPat[pattern][frame].freqs = make([]float64, nfreqs)
							spcc.speechPat[pattern][frame].amps = make([]float64, nfreqs)
							for i := 0; i < nfreqs; i++ {
								freq, err := strconv.ParseFloat(items[i], 64)
								if err != nil {
									plot.Status = fmt.Sprintf("freq %d conversion error: %v", i, err.Error())
									// Write to HTTP using template and grid
									if err := tmplTestingSPCC.Execute(w, plot); err != nil {
										log.Fatalf("Write to HTTP output using template with error: %v\n", err)
									}
									return
								}
								ampl, err := strconv.ParseFloat(items[i+nfreqs], 64)
								if err != nil {
									plot.Status = fmt.Sprintf("ampl %d conversion error: %v", i, err.Error())
									// Write to HTTP using template and grid
									if err := tmplTestingSPCC.Execute(w, plot); err != nil {
										log.Fatalf("Write to HTTP output using template with error: %v\n", err)
									}
									return
								}
								spcc.speechPat[pattern][frame].amps[i] = ampl
								spcc.speechPat[pattern][frame].freqs[i] = freq
							}
							frame++
						}
						fspeech.Close()
						if err = scanner.Err(); err != nil {
							fmt.Printf("speech file scanner error: %s", err.Error())
							// Write to HTTP using template and grid
							if err := tmplTestingSPCC.Execute(w, plot); err != nil {
								log.Fatalf("Write to HTTP output using template with error: %v\n", err)
							}
							return
						}
						pattern++
					}
				}
			}
		}

		// generate speech, spectrogram if desired, create wav file,
		// using speech pattern, percent voiced, deltas duration, amplitude, and pitch

		err = spcc.createSpeech(speechPattern)
		for err != nil {
			if err.Error() == "repeat" {
				err = spcc.createSpeech(speechPattern)
			} else {
				fmt.Printf("createSpeech error: %v\n", err.Error())
				plot.Status = fmt.Sprintf("createSpeech error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplaySPCC.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
		}

		if spcc.domain == "spectrogram" {
			spcc.grayscale = make(map[int]string)
			for i := 0; i < ncolors; i++ {
				spcc.grayscale[i] = fmt.Sprintf("gs%d", i)
			}

			err := spcc.processSpectrogram(speechPattern, spcc.fftWindow, spcc.fftSize)
			if err != nil {
				fmt.Printf("proessSpectrogram error: %v\n", err)
				plot.Status = fmt.Sprintf("processSpectrogram error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplaySPCC.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status = fmt.Sprintf("Spectrogram of pattern %d plotted.", speechPattern)
		} else {
			err := spcc.processTimeDomain(speechPattern)
			if err != nil {
				fmt.Printf("processTimeDomain error: %v\n", err)
				plot.Status = fmt.Sprintf("processTimeDomain error: %v", err.Error())
				// Write to HTTP using template and grid
				if err := tmplDisplaySPCC.Execute(w, plot); err != nil {
					log.Fatalf("Write to HTTP output using template with error: %v\n", err)
				}
				return
			}
			plot.Status = fmt.Sprintf("Time Domain of pattern %d plotted.", speechPattern)
		}

		// Create the wav file from the synthetic speech
		outF, err := os.Create(path.Join(dataDir, synSpeech))
		if err != nil {
			fmt.Printf("os.Create() file %s error: %v\n", synSpeech, err)
			plot.Status = fmt.Sprintf("os.Create() file %s error: %v", synSpeech, err.Error())
			// Write to HTTP using template and grid
			if err := tmplDisplaySPCC.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		defer outF.Close()
		// create wav.Encoder
		enc := wav.NewEncoder(outF, sampleRate, bitDepth, 1, 1)

		// create audio.FloatBuffer
		float64Buf := &audio.FloatBuffer{Data: spcc.synSpeech, Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate}}

		// create IntBuffer from FloatBuffer and pass to Encoder.Write()
		if err := enc.Write(float64Buf.AsIntBuffer()); err != nil {
			fmt.Printf("wav encoder write error: %v\n", err)
			plot.Status = fmt.Sprintf("wav encoder write error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplDisplaySPCC.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// close the encoder
		if err := enc.Close(); err != nil {
			fmt.Printf("wav encoder close error: %v\n", err)
			plot.Status = fmt.Sprintf("wav encoder close error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplDisplaySPCC.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Play the audio wav if fmedia is available in the PATH environment variable
		fmedia, err := exec.LookPath("fmedia.exe")
		if err != nil {
			log.Fatal("fmedia is not available in PATH")
		} else {
			fmt.Printf("fmedia is available in path: %s\n", fmedia)
			cmd := exec.Command(fmedia, filepath.Join(dataDir, synSpeech))
			stdoutStderr, err := cmd.CombinedOutput()
			if err != nil {
				fmt.Printf("stdout, stderr error from running fmedia: %v\n", err)
			} else {
				fmt.Printf("fmedia output: %s\n", string(stdoutStderr))
			}
		}

		// set the speech parameters
		spcc.plot.DelDuration = strconv.Itoa(spcc.delDuration)
		spcc.plot.DelPitch = strconv.Itoa(spcc.delPitch)
		spcc.plot.DelAmpl = strconv.FormatFloat(spcc.delAmpl, 'f', -1, 64)
		spcc.plot.PercentVoiced = strconv.Itoa(spcc.percentVoiced)
		spcc.plot.SpeechPattern = strconv.Itoa(speechPattern)

		// Execute plot on display HTML template
		if err = tmplDisplaySPCC.Execute(w, spcc.plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	} else {
		plot.Status = "Enter Display speech parameters:  pattern, percent voiced, pitch delta, duration delta."
		// Write to HTTP using template and grid
		if err := tmplDisplaySPCC.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}
}

// executive creates the HTTP handlers, listens and serves
func main() {
	// Set up HTTP servers with handlers for testing the Spectrogram Cross Correlation

	// Create HTTP handler for testing
	http.HandleFunc(patternTestingSPCC, handleSPCC)
	// Create HTTP handler for display of synthetic speech in time/amplitude or spectorgram
	http.HandleFunc(patternDisplaySPCC, handleDisplaySPCC)
	fmt.Printf("Speech Sectrogram Cross Correlation Server listening on %v.\n", addr)
	http.ListenAndServe(addr, nil)
}
