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

Finally the logger information can be limited to some detectors only using a key,value map file.

```bash
MCSTEPLOG_VOLMAPFILE=path_to_/volmapfile.dat MCSTEPLOG_TTREE=1 LD_PRELOAD=path_to/libMCStepLogger.so o2sim ..
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



