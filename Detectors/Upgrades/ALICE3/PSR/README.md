<!-- doxy
\page refDetectorsUpgradesALICE3PSR Preshower
/doxy -->

# PostLS4Preshower

This is a simple 8 layered preshower detector named PSR. This is a cylindrical detector of length 100 cm whose shower layers are made up of Pb (0.5 cm) and the detector layers are made up of Si (45 microns). PSR is based on the ITSMFT classes and the code is structurally similar to that of FT3. Each layer is made of a monolithic silicon disk with a thin sensitive layer for hit generation. Silicon chip thickness is tuned to match the layer x/X0 to allow a minimal evaluation of material budget effects.

One should get a file o2sim_HitsPSR.root by running
$ o2-sim -m PSR -e TGeant3 -g boxgen -n 10

<!-- doxy
/doxy -->
