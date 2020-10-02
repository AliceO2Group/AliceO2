<!-- doxy
\page refGeneratorsshareexternal External Generators
/doxy -->

# External generators


------------


## QEDLoader.C / QEDepem.C

-  Invokes TGenEpEmv1 generator from [AEGIS](https://github.com/AliceO2Group/AEGIS) package for Pb-Pb &rarr;  e+e- generation.
	+	optional parameters are rapidity and pT ranges to generate, can be provided as key/value option, e.g.
``
o2-sim -n 1000 -m PIPE ITS -g external --configKeyValues 'GeneratorExternal.fileName=$O2_ROOT/share/Generators/external/QEDLoader.C;QEDGenParam.yMin=-3;QEDGenParam.ptMin=0.001'
``
The x-section of the process depends on the applied cuts, it is calculated on the fly and stored in the ``qedgenparam.ini`` file.

## GenCosmicsLoader.C / GenCosmics.C

-  Invokes GenerateCosmics generators from [AEGIS](https://github.com/AliceO2Group/AEGIS) package.

``o2-sim -n1000 -m PIPE ITS TPC -g extgen --configKeyValues "GeneratorExternal.fileName=$O2_ROOT/share/Generators/external/GenCosmicsLoader.C"``

Generation options can be changed by providing ``--configKeyValues "cosmics.maxAngle=30.;cosmics.accept=ITS0"`` etc.
For instance, to generate track defined at radius 500 cm, with maximal angle wrt the azimuth of 40 degress and passing via ITS layer 0 at Y=0:

``o2-sim -n100 -m PIPE ITS TPC -g extgen --configKeyValues "cosmics.maxAngle=40.;cosmics.accept=ITS0;cosmics.origin=500;GeneratorExternal.fileName=$O2_ROOT/share/Generators/external/GenCosmicsLoader.C"``

See GenCosmicsParam class for available options.
------------
