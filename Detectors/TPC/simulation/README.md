<!-- doxy
\page refTPCsimulation TPC simulation
/doxy -->

# TPC Digitization

## Outline

The TPC digitization handles the transformation from the simulated GEANT hits to the output of the front-end electronics.
The workflow and the validation of the corresponding output is described in detail in the [PhD thesis of Andi Mathis](http://cds.cern.ch/record/2728287). In the following, a concise overview is presented.
For the digitization as conducted by the [TPC digitizer](include/TPCSimulation/Digitizer.h) the following physics processes are considered:

* The energy loss of each individual GEANT hit is converted into a number of electrons by dividing by the effective ionization potential W_i. Each of these electrons is in the following treated individually.
* The electron is projected onto the readout plane, taking into account diffusion, i.e. smearing its position by a 3D gaussian function ([ElectronTransport](include/TPCSimulation/ElectronTransport.h)). Then, the position is transformed into the local coordinate system of the Readout Chamber (ROC).
* Having arrived at the amplification stage, the electrons undergo amplification in the GEM stack ([GEMAmplification](include/TPCSimulation/GEMAmplification.h)), taking into account fluctuations of the gain. These fluctuations follow a Polya distribution. For performance considerations, two different versions of the amplification are available (one effective single-stage amplification and a successive simulation of the collection, amplification, and extraction processes in the individual GEMs.
* Capacitive coupling of the amplification structure to the readout anode leads to another contribution to the signal, the so-called Common Mode effect. Since the bottom electrode of GEM 4 is unsegmented, capacitive coupling occurs within a full ROC ([CommonMode](include/TPCSimulation/CommonMode.h))
* The charge signal is then folded with the transfer function of the front-end cards ([SAMPAProcessing](include/TPCSimulation/SAMPAProcessing.h)) and written to the intermediate storage container structure ([DigitContainer](include/TPCSimulation/DigitContainer.h)/[DigitTime](include/TPCSimulation/DigitTime.h)/[DigitGlobalPad](include/TPCSimulation/DigitGlobalPad.h)), which is described below.

## Workflow

The digitization is conducted for each TPC sector individually in order to ensure maximal parallelization. The workflow is implemented in the DPL and the `o2-sim-digitizer-workflow`.

### Input data
The input can be created by running the simulation `o2-sim`, which produces the file `o2sim.root` with the hits stored in separated branches for all sectors.
It should be noted that due to diffusion and the space-charge distortions, charge leakage between sectors can occur. In order to avoid the unnecessary processing of individual hits, several measures are taken

* for a given sector the hits within an additional safety margin of +/- 10 degree are processed. For this reason, For this reason, the hits are not stored for a given sector, but shifted by 10 degrees. Hence only two branches need to be loaded for the digitization of a given sector.
* Individual hits are only processed when they are within the processed by 3 sigma of the expected width from diffusion

Hits passing that requirement are further processed by the [TPC digitizer](include/TPCSimulation/Digitizer.h) and undergo the above described physics processes.

### Intermediate container structure

In particular in the case of large track densities, the hits from different tracks can contribute to an individual digit (pad, row, time bin). Therefore, an intermediate buffering of the digits is necessary. This is accomplished by an intermediate container structure.
The [DigitContainer](include/TPCSimulation/DigitContainer.h) is a circular buffer of time bins, in order to enable continuous readout of the detector as explained below.
The [DigitTime](include/TPCSimulation/DigitTime.h) is then a flat contained of [DigitGlobalPad](include/TPCSimulation/DigitGlobalPad.h), where the latter correspond to one pad on the pad plane. Accordingly, the buffering of the actual ADC values is conducted using this object.
Similarly, the MC labels are passed throughout the chain, and finally sorted by the number of occurrences, i.e. the track with the largest contribution to the digit is mentioned first etc.

Correlations among digits from different events can only occur within the integration time of the detector (plus additional 50% contigiency), and therefore the digits are written to disk when the processed event is more than 750 time bins after. This means, that saturation effects are applied to the ADC values stored in the [DigitGlobalPad](include/TPCSimulation/DigitGlobalPad.h) and the relevant information is transformed in a [Digit](../../../DataFormats/Detectors/TPC/include/DataFormatsTPC/Digit.h) which is written to disk.

### Output data
The digitizer workflow produces the file `tpcdigits.root` by default, data is stored in separated branches for all sectors.