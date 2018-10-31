set(MODULE_NAME "ITSMFTReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/ChipMappingITS.cxx
  src/ChipMappingMFT.cxx
  src/DigitPixelReader.cxx	
  src/Clusterer.cxx
  src/PixelData.cxx
  src/BuildTopologyDictionary.cxx
  src/LookUp.cxx
  src/TopologyFastSimulation.cxx
  src/AlpideCoder.cxx
)
set(HEADERS
  include/${MODULE_NAME}/PixelReader.h
  include/${MODULE_NAME}/DigitPixelReader.h
  include/${MODULE_NAME}/RawPixelReader.h
  include/${MODULE_NAME}/PixelData.h
  include/${MODULE_NAME}/Clusterer.h
  include/${MODULE_NAME}/BuildTopologyDictionary.h
  include/${MODULE_NAME}/LookUp.h
  include/${MODULE_NAME}/TopologyFastSimulation.h
  include/${MODULE_NAME}/ChipMappingITS.h
  include/${MODULE_NAME}/ChipMappingMFT.h
  include/${MODULE_NAME}/AlpideCoder.h
)
Set(LINKDEF src/ITSMFTReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME itsmft_reconstruction_bucket)
O2_GENERATE_LIBRARY()
