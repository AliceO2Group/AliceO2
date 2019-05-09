\page refTRDbase TRD base

This is a short documention for the TRD base software for O2

At the moment this is just a poc for doing documentation.
# Status/Description of implementation


* ArrayADC:
  - Store the raw adc values in a linearised 2d array.
  - 30 time bins (1 "axis) for each pad
  - all 144/168 pads
  - the 144 physical pads are mapped to MCM, with overlaps explaining 168.
  - the mapping is a lookup table stored in FeeParams (front end electronics parameters)
  
* ArraySignal:
  -  similar arrayADC.
  - 
* ArrayDictionary:
  - similar arrayADC.
  -
* FeeParam:
  - singleton
  - frontend electronics paramaters.
  - lookup array for mapping pads to MCM, the Array[???] used to each contain this.
* MCMSimulator:
  - software implementation of the MCM logic in the ASIC chips for purposes of simulation.
* SignalIndex 
  - Index into arrayADC.
  - Sparsematrix version of ArrayADC will negate this class
* TRDCalChamberStatus

# Missing things/Improvements to come
 
* Change the underlying storage with in the array to a sparse structure.
   - the Arrays have a 5% occupancy for min bias PbPb [citation needed]
