\page refMUONMIDWorkflow MID Workflow

# Running MID workflow

The MID clustering takes as input the "digits" produced by the simulation and returns the recontructed tracks and the associated MC labels.

## Getting the digits
If you do not have the digits, you can obtain a sample with:
```bash
o2-sim -g fwmugen -m MID -n 100
o2-sim-digitizer-workflow
```

## Execution
To run the reconstruction workflow, run:
```bash
o2-mid-reco-workflow
```

Options:

`--disable-mc`
:  do not propagate the MC info (faster: this is what happens with real data)

### CAVEATs

*   The termination of the workflow is currently tricky. So this workflow does not automatically quit when the processing is done. The program should be therefore either quit manually or one needs to specify a number of events to analyse with option: `--nevents <number_of_events>`
*   Also, the reconstruction part is way faster than the label propagation part. So it can happen that the MC labels are dropped before being processed. This issue should be solved at the level of DPL. In the meanwhile a temporary solution consists in changing the calue of DEFAULT_PIPELINE_LENGTH in the  [DataRelayer](../../../../Framework/Core/src/DataRelayer.cxx). A value of 64 was found to be enough.