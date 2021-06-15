#! /bin/bash

o2-sim -n 10 -g pythia8pp --skipModules ZDC > o2sim.log
o2-sim-digitizer-workflow -b --onlyDET TRD > o2digitizer.log
o2-trd-trap-sim -b >trapsim.log
