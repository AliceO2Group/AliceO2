## MCStepLogger

Detailed debug information about stepping can be directed to standard output using the `LD_PRELOAD` env variable, which "injects" a
 special logging library (which intercepts some calls) in the executable that follows in the command line.

```bash
LD_PRELOAD=path_to/libMCStepLogger.so o2sim -m MCH -n 10
```


```
[MCLOGGER:] START FLUSHING ----
[STEPLOGGER]: did 28 steps
[STEPLOGGER]: transported 1 different tracks
[STEPLOGGER]: transported 1 different types
[STEPLOGGER]: VolName cave COUNT 23 SECONDARIES 0
[STEPLOGGER]: VolName normalPCB1 COUNT 3 SECONDARIES 0
[STEPLOGGER]: ----- END OF EVENT ------
[FIELDLOGGER]: did 21 steps
[FIELDLOGGER]: VolName cave COUNT 20
[FIELDLOGGER]: ----- END OF EVENT ------
[MCLOGGER:] END FLUSHING ----
```

The stepping logger information can also be directed to an output tree for more detailed investigations.
Default name is `MCStepLoggerOutput.root` (and can be changed
by setting the `MCSTEPLOG_OUTFILE` env variable).

```bash
MCSTEPLOG_TTREE=1 LD_PRELOAD=path_to/libMCStepLogger.so o2sim ..
```

Finally the logger can use a map file to give names to some logical grouping of volumes. For instance to map all sensitive volumes from a given detector `DET` to a common label `DET`. That label can then be used to query information about the detector steps "as a whole" when using the `StepLoggerTree` output tree.

```bash
> cat volmapfile.dat
normalPCB1 MCH
normalPCB2 MCH
normalPCB3 MCH
normalPCB4 MCH
normalPCB5 MCH
normalPCB6 MCH
centralPCB MCH
downroundedPCB MCH
uproundedPCB MCH
cave TheCavern

> MCSTEPLOG_VOLMAPFILE=path_to_/volmapfile.dat MCSTEPLOG_TTREE=1 LD_PRELOAD=path_to/libMCStepLogger.so o2sim ..

> root -b MCStepLoggerOutput.root
root[0] StepLoggerTree->Draw("Lookups.volidtomodule.data()");
```

Note also the existence of the `LD_DEBUG` variable which can be used to see in details what libraries are loaded (and much more if needed...).

```bash
LD_DEBUG=libs o2sim
LD_DEBUG=help o2sim
```

## Special case on macOS

`LD_PRELOAD` must be replaced by `DYLD_INSERT_LIBRARIES`, e.g. :

```bash
DYLD_INSERT_LIBRARIES=/Users/laurent/alice/sw/osx_x86-64/O2/latest-clion-o2/lib/libMCStepLogger.dylib MCSTEPLOG_TTREE=1 MCSTEPLOG_OUTFILE=toto.root o2sim -m MCH -g mugen -n 1
```

`LD_DEBUG=libs` must be replaced by `DYLD_PRINT_LIBRARIES=1`

`LD_DEBUG=statistics` must be replaced by `DYLD_PRINT_STATISTICS=1`


## MCStepLogAnalysis

Information collected and stored in `MCStepLoggerOutput.root` can be further investigated using the excutable `mcStepAnalysis`. This executable is independent of the simulation itself and produces therefore no overhead when running a simulation. 2 commands are so far available (`analyze`, `checkFile`) including useful help message when typing
```bash
mcStepAnalysis <command> --help
```

### File formats

2 file formats having a standardised structure play a role. On one hand, these are the files produced by the step logging which are the input files for the analysis as explained in the following. On the other hand, each analysis produces an output file containing histograms along with some meta information. Sanity and the type of the file can be checked via
```bash
mcStepAnalysis checkFile -f <FileToBeChecked>
```
### Analysing the steps

The basic command containing all required parameters is
```bash
mcStepAnalysis analyze -f <MCStepLoggerOutputFile> -o <parent/output/dir> -l <label>
```
where  
* `-f <MCStepLoggerOutputFile>` passes the input file produced with the `MCStepLogger` as explained above (default name is `MCStepLoggerOutput.root`)
* `-o <parent/output/dir>` provides the top directory for the analysis output (if this does not exist, it is created automatically)
* `-l <label>` adds a label, e.g. for plots produced later.

A `ROOT` file at `parent/output/dir/MetaAnalysis/Analysis.root` is produced containing all histograms as well as important meta information. Histogram objects are derived from `ROOT`s `TH1` classes.

### Further processing of analysis files

Files produced as described before can be investigated further or used to plot the histograms therein. The interface to read these files is the class `AnalysisFile` and histograms can be requested by their names.
```c++
// somthing...
#include "MCStepLogger/AnalysisFile.h"
// some code
AnalysisFile af;
af.read("path/to/file.root");
// print meta info
af.printMetaInfo();
// get a histogram by name (program will exit if name not found)
TH1& histogram = af.getHistogram("nameOfHistogram");
// if you know the underlying object is, e.g. of derived class TH1D and you really need that you can do
TH1D& otherHistogramCasted = af.getHistogram<TH1D>("nameOfOtherHistogram");
// where the program will safely exit immediately in case the casting was not successful
//...
// modify histogram or do other things
//...
//to save changes made to histograms do
af.write("path/to/output/file.root");
```
### Additional information about the `MCAnalysisManager`

There is the staic method `MCAnalysisManager::Instance()` which returns a reference to a static instance. So always make sure you don't copy it but get the reference in case you want to work with that instance on a global scope, i.e.
```c++
// some code
auto& anamgr = MCAnalysisManager::Instance();
```
### Additional information about the analysis objects

Histograms which should be written to disk in an analysis are managed by `MCAnalysisFileWrapper` objects. These also make sure that no histogram is created twice. Therefore, all of these histograms should be created like `T* myHisto = MCAnalysis::getHistogram<T>(...)` where the template parameter `T` must be a class deriving from ROOT's `TH1`. It then returns a pointer to the desired object. Managing histograms not on the level of an analysis also enables for requesting histograms from another analysis. In that way one can write a custom analysis for a specific use case but can still ask for e.g. for a histogram from the `BasicMCAnalysis` to derive some additional and more generic information about a simulation run. Hence, never manually delete an object obtained like this.

### Comparing analysis values to reference reference values

For the `BasicMCAnalysis` there is a small test suite to compare the obtained values from a simulation run to reference values contained in a JSON file. So far, that is a prototype only caring about the total number of steps and total number of tracks obtained in the simulation. The `JSON` file looks as follows
```json
{
  "analysisName": "BasicMCAnalysis",
  "nSteps": [absoluteNumberOfSteps, relativeTolerance],
  "nTracks": [absoluteNumberOfTracks, relativeTolerance]
}
```
The test is steered via
```bash
runTestBasicMCAnalysis -- <path/to/Analysis.root> <reference.json>
```
Note, that the test does not know anything about the settings of the simulation run, i.e. there are no information about the primary generator of the transport engine etc. The user has to make sure to apply this coherently.

### Writing/running a custom analysis

Although providing already a number of different observables, users might want to add custom observables for their analysis. To do so, a directory for custom analyses has to be created where analysis macros can be provided and loaded at run-time. Note, that only the basic analysis is actually contained in the compiled code. One of the main reasons for that is to enable for a coherent comparison between different points in the git history. However, if you feel like there is an important observable missing, feel free to report that.

The logic of adding a custom analysis is very similar to that of `Rivet` and the general workflow should look familiar in any case. Say, your analysis macro directory is `$ANALYSIS_MACROS/` where you have your macro `mySimulationAnalysisc.C` (don't place any other files there since these cannot be read...). A skeleton looks as follows, also containing more information on how and why things are implemented like they are:

```c++
// myMCStepAnalysis.C
//
// headers you need

/* Your analysis class has to derive from the base o2::mcstepanalysis::MCAnalysis. All
 * methods to be implemented are mentioned below.
 */
class MyMCStepAnalysis : public MCAnalysis
{
  public:
  	/* You don't need a fancy constructor, that's it. The base constructor takes care
  	 * of registering this analysis to the global o2::MCStepAnalysis::MCAnalysisManager
  	 * object. An MCAnalysis object cannot exist without the MCAnalysisManager.
  	 */
    MySimulationAnalysis()
      : MCAnalysis("MyMCStepAnalysis")
    {}

    /* There are 3 methods covering the analysis: initialize, analyze and finalize.
     * The first two need to be overriden in any case.
     * These methods are called by the MCAnalysisManager. Other custom methods
     * can hence only be used for class-internal purposes only.
     */

    /* All histograms used in the analysis must be defined here. This is done using
     * the method MCAnalysis::getHistogram<T>(...) where the template paramter T has to
     * be a deriving class of ROOT's TH1. The histogram pointers are the managed
     * globally for further processing, saving, plotting etc. It is also taken care
     * of the deletion of the pointers. Do not attempt to delete histograms obtained by
     * MCAnalysis::getHistogram<T>(...). Histograms are uniquely identified by there
     * name and an MCAnalysis will stop immediately at the initialisation stage if
     * two histograms with the same name are detected.
     */
    void initialize() override {

    	// monitor calls to the magnetic field per volume
    	histMyFirstObservable = getHistogram<TH1D>("myFirstObservable", 1, 0., 1.);
    	// monitor the steps in the r-z plane
    	histMySecondObservable = getHistogram<TH2D>("mySecondObservable", 1, 0., 1., 2, 0., 2.);
    	// more histograms?! Other things to do?

    }

    /* This method is used to extract the step information in order to fill histograms
     * accordingly.
     */
    void analyze(const std::vector<StepInfo>* const steps,
    			       const std::vector<MagCallInfo>* const magCalls) override {

      // loop over mag field calls and match to step ID
    	for(const auto& call : *magCalls) {
		    auto step = steps->operator[](call.stepid);
		    mAnalysisManager->getLookupVolName(step.volId, volName);
		    histMyFirstObservable->Fill(volName.c_str(), 1.);
	    }

      // loop over steps and get z-position and other things as you like
	    for(const auto& step : *steps) {
	    	histMySecondObservable->Fill(step.z, std::sqrt( step.x*step.x + step.y*step.y ));
	    	// ...and do more, as you like
	    }
    }

    /* the finalyze method can be used to do necessary adjustments like scaling or adding
     * histograms or whatever must be done to them.
     */
    void finalize() override {
    	histMyFirstObservable->Scale( 1. / histMyFirstObservable->GetEntries() );
    	// ...and more...
    }

  private:
  	// you might want to provide some useful methods for internal use...

  private:
  	TH1D* histMyFirstObservable;
  	TH2D* histMySecondObservable;
  	// other histogram pointers and/or members for internal usage
 };

/* The hook to bring your analysis into existence so that the MCAnalysisManager
 * knows. Don't be scared about the way the pointer of the analysis is created. It is all
 * taken care of by the MCAnalysisManager since the base MCAnalysis constructor
 * automatically registeres itself to the MCAnalysisManager. So don't worry about
 * pointers flying around.
 */
void declareAnalysis() {
	new MySimulationAnalysis("mySimulationAnalysis");
}
```
After having this, you are ready to include this in the analysis run by typing
```bash
runMCAnalysis analyze -f <MCStepLoggerOutputFile> -o <parent/output/dir> -l <label> -d $ANALYSIS_MACROS -a mySimulationAnalysis
```
where now
* `-d $ANALYSIS_MACROS` points the executable to the directory of where your macros are located
* `-a  mySimulationAnalysis` tells which analysis to load. In case you have more analyses in that directory you want to load, just append the names of all analyses you want to run.
The output of the custom analysis is written to `parent/output/dir/mySimulationAnalysis/` and that's it.
