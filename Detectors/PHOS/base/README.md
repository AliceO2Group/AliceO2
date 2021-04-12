<!-- doxy
\page refDetectorsPHOSGeometry PHOS Geometry
/doxy -->

# PHOS geometry

Module numbering:  start from module 0 (non-existing), 1 (half-module), 2 (bottom),... 4(highest)

All channels have unique absId: start from 1 till 4*64*56. Numbering in each module starts at bottom left and first go in z direction:
  56   112   3584
  ...  ...    ...
  1    57 ...3529

One can use also relative numbering  relid[3]: 
(module number[0...3], iphi[1...64], iz[1...56])

  Then TRU channels go 112 per branch, 2 branches per ddl
tru channels have absId after readout channels:
  absId = getTotalNCells() + TRUabsId ;
  relId for TRU
  relid: [DDL id=0..13] [x in 2x2 system: 0..7] [z in 2x2 system 0..27] 

Mapping is realized with class Mapping and mapping files are stored as MODxRCUy.data  

<!-- doxy
/doxy -->