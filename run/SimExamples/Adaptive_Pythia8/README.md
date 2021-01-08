<!-- doxy
\page refrunSimExamplesAdaptive_Pythia8 Example Adaptive_Pythia8
/doxy -->

This is a simple simulation example showing

a) how to run a simple background event simulation with some parameter customization
b) how to run a simulation producing signal events based on a custom external generator that can adapt its behaviour depending on the characteristics of the background event.

Pythia8 events are generated according to the configuration given in the file `adaptive_pythia8.macro`.
The settings are as such that Pythia8 is initialised using the `pythia8_inel.cfg` configuration file.
The customisation allows the generator to receive and react to a notification that signals the embedding status of the simulation, giving the header of the background event for determination of the subsequent actions. In this case, the number of pythia8 events to be embedded is calculated according to a formula that uses the number of primary particles of the background events.

The macro file is specified via `--configKeyValues` setting `GeneratorExternal.fileName` whereas the specific function call to retrieve the configuration and define the formula is specified via `GeneratorExternal.funcName`.
