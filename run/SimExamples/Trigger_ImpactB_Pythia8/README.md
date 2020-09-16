<!-- doxy
\page refrunSimExamplesTrigger_ImpactB_Pythia8 Example Trigger_ImpactB_Pythia8
/doxy -->

This is a simple simulation example showing how to run event simulation using the Pythia8 event generator in heavy-ion mode and use an external custom trigger.
The external trigger makes use of the `DeepTrigger` functionality to allow the user to access the internals of the event generator to define the trigger.
In this example, information about the impact parameter of the currently-generated Pythia8 heavy-ion event is used.
The trigger selection defines impact parameters to be accepted according to a min-max range that can be adjusted from the command line.

The definition and configuration of the external `DeepTrigger` is performed by the function `trigger_impactb_pythia8(double bMin = 0., double bMax = 20.)` defined in the macro `trigger_impactb_pythia8.macro`.

The macro file and function names are specified via `--configKeyValues` setting `TriggerExternal.fileName` and `TriggerExternal.funcName`, respectively.

# WARNING
Using a trigger is not always the most efficient way to deal with event generation.
It is always advisable to find a way to bias the event-geneator process to save CPU time.
In this case, the selection of a specific impact-parameter range in Pythia8 can lead to
a large time spent in the event generation rather than in the simulation.
