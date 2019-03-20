#!/bin/bash
o2sim -n 100 -e TGeant3 -g boxgen -m PIPE ITS >& sim.log 
digitizer-workflow
qcRunPrint


