<!-- doxy
\page refrunSimExamplesSignal_ImpactB Example Signal_ImpactB
/doxy -->

This is a simulation example showing the following things

a) how to run a simple background event simulation with some parameter customization
b) how to setup and run an event generator that produces signal events based on the impact parameter of the backround event where it will be embetted into

Custom signal events generated according to the configuration given in a file 'signal_impactb.macro'.
The custom event generator receives and react to a notification that signals the embedding status of the simulation, giving the header of the background event for determination of subsequent actions.
In this case, the impact paramereter from the background event is used to calculate the number of particles to be generated as signal

The macro file is specified via `--configKeyValues` setting `GeneratorExternal.funcName` whereas the specific function call to retrieve the configuration and define the formula is specified via `GeneratorExternal.funcName`.

The event generator for the background embedding needs to be capable of providing an impact parameter. In this case, Pythia8 heavy-ion model provides such value.
