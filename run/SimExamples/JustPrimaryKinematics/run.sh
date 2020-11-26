# A very simple example, showing how to configure o2-sim to skip
# transport/physics and save just the originally generated event.
# While this can be achieved in potentially many ways, the example is instructive
# to demonstrate the configuration mechanism in general.
# Here the mechanism is to switch off physics in the configuration file and set very
# tight geometry cuts so that Geant will not do work.

o2-sim -n 10 -g pythia8 -m CAVE --configFile only_kine.ini
