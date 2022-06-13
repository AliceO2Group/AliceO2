<!-- doxy
\page refDetectorsUpgradesALICE3FCT EndCaps
/doxy -->

# Forward Conversion Tracker

This is a top page for the Forward Conversion Tracker (FCT) detector documentation.

The FCT measures photon conversions at forward rapidity, 3 < $\eta$ < 5. 

Simulations can be run with three options
1. Default version. This builds the FCT according to thhe function detector::buildFCTV1. No parameters are required
2. Using basic parameters. This builds the FCT according to the parameters passed to FCTBaseParam
3. Using a config file. This builds the FCT from a config file. To do this, use 
```
o2-sim -m FCT -e TGeant3 -g boxgen -n 10 --configKeyValues 'BoxGun.pdg=13 ; BoxGun.eta[0]=-5.0 ; BoxGun.eta[1]=-3.0; BoxGun.number=500; FCTBase.configFile=my_fwd_detector.cfg'
```
An example for the config file can be found at https://github.com/Cas1997/O2_ALICE_3_Example_files/blob/main/FCT_layout_example.cfg

Digitization is not implemented as of yet


<!-- doxy
/doxy -->
