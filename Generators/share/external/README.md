# External generators


------------


## qedbkg.C

-  Invokes TGenEpEmv1 generator from [AEGIS](https://github.com/AliceO2Group/AEGIS) package for Pb-Pb &rarr;  e+e- generation.
	+	optional parameters are rapidity and pT ranges to generate, can be provided as key/value option, e.g.
``
o2-sim -n 1000 -m PIPE ITS -g extgen --extGenFile $O2_ROOT/share/Generators/external/QEDLoader.C  --configKeyValues 'QEDGenParam.yMin=-3;QEDGenParam.ptMin=0.001'
``
The x-section of the process depends on the applied cuts, it is calculated on the fly and stored in the ``qedgenparam.ini`` file.

------------
