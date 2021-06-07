#!/bin/bash

# make sure the steplogger environment is loaded:
# alienv enter O2/latest MCStepLogger/latest

# This demonstrates the most basic usage of the steplogger
LD_PRELOAD=$MCSTEPLOGGER_ROOT/lib/libMCStepLoggerIntercept.so o2-sim-serial -n 1 -g pythia8pp -m PIPE ITS TPC
