# A very simple example, showing how to configure o2-sim to skip
# transport/physics and save just the originally generated event.
# While this can be achieved in potentially many ways, the example is instructive
# to demonstrate the configuration mechanism in general.
# Here the mechanism is to switch off physics in the configuration file and set very
# tight geometry cuts so that Geant will not do work.

# first stage --> produce events using Pythia8 (no transport)
o2-sim -n 10 -g pythia8pp -m CAVE --configFile only_primarykine.ini

# second stage --> read back events from O2 kine (no transport)
o2-sim -n 10 -g extkinO2 --extKinFile o2sim_Kine.root -m CAVE --configFile only_primarykine.ini -o o2sim2

# ideally here, both kinematics files should be identical
