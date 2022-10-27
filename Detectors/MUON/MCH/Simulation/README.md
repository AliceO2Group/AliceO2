# How to make a simulation including MCH

## Only MCH

Like all [O2 simulations](../../../../doc/DetectorSimulation.md), it's a two stage process : first create the hits and then digitize them.

### Hit creation

     o2-sim -n 10 -g fwmugen -e TGeant3 -m MCH

will generate 10 events using the `fwmugen` generator (1 forward muon per event), using Geant3 as transport, and with only MCH geometry.

     CAUTION: by specifying -m MCH you only get MCH geometry, no absorber, no dipole, no beam shield, etc... So that's only a relevant option for quick debug, not for any kind of physics analysis.

This steps creates (among others) an `o2sim_HitsMCH.root` file with the MCH hits, as well as `o2sim_geometry.root` and `o2sim_grp.root` which are used later on by the digitizer.

### Digitization

In the same directory where the hits were created, use:

     o2-sim-digitizer-workflow

This will creates `mchdigits.root`, updates `o2sim_grp.root` and generates a `collisioncontext.root`.
This last file can be used to redo the digitization for the very _same_ collisions, but with e.g. different digitization parameters.

Various options can be specified using the `--configKeyValues`.
For example for the MCH digitizer:

     o2-sim-digitizer-workflow --configKeyValues "MCHDigitizer.noiseOnlyProba=1.e-6;MCHDigitizer.timeSigma=6."

For the list of possible parameters see the [DigitizerParam](./include/MCHSimulation/DigitizerParam.h) and [ResponseParam](../Base/include/MCHBase/ResponseParam.h) structs.

## More realistic simulations

For anything but debug, you should include more modules at the hit creation stage:

     o2-sim -m HALL MAG DIPO COMP PIPE ABSO SHIL MCH

Of course, add other detectors (MID,MFT,ITS,...) if need be.

Note that there is a way to simply include the geometry of a detector, just to have its material budget in the picture, without also having it participating to the hit creation, see PR: https://github.com/AliceO2Group/AliceO2/pull/8149 for the options.
