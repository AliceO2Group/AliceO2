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



