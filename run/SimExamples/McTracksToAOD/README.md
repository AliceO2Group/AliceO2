<!-- doxy
\page refrunSimExamplesMcTracksToAOD Example McTracksToAOD
/doxy -->

These are examples demonstrating how to do on-the-fly event generation
for DPL (analysis) tasks.

List of examples:
- run.sh --> inject events from existing kinematics file
- run_Pythia8.sh --> generate Pythia8 events in DPL device and forward to analysis
- run_trigger.sh --> generate Pythia8 events with triggering in DPL device and forward to analysis
- run_O2Kine.sh  --> generate events and save them in kinematics file; read back events and publish to analysis task