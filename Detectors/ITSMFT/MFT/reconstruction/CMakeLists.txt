set(MODULE_NAME "MFTReconstruction")

O2_SETUP(NAME ${MODULE_NAME})

set(SRCS
  src/Cluster.cxx
  src/ClusterizerTask.cxx
  src/Clusterizer.cxx
  src/Track.cxx
  src/FindTracks.cxx
)
set(HEADERS
  include/${MODULE_NAME}/Cluster.h
  include/${MODULE_NAME}/ClusterizerTask.h
  include/${MODULE_NAME}/Clusterizer.h
  include/${MODULE_NAME}/Track.h
  include/${MODULE_NAME}/FindTracks.h
)

set(NO_DICT_SRCS
  src/devices/Sampler.cxx
  src/devices/FileSink.cxx
  src/devices/Merger.cxx
)

set(NO_DICT_HEADERS
  include/${MODULE_NAME}/devices/Sampler.h
  include/${MODULE_NAME}/devices/TaskProcessor.h
  include/${MODULE_NAME}/devices/TaskProcessor.tpl
  include/${MODULE_NAME}/devices/FileSink.h
  include/${MODULE_NAME}/devices/Merger.h
)

Set(LINKDEF src/MFTReconstructionLinkDef.h)
Set(LIBRARY_NAME ${MODULE_NAME})
Set(BUCKET_NAME mft_reconstruction_bucket)

O2_GENERATE_LIBRARY()

Set(Exe_Names
  mft-reco-sampler
  mft-reco-processor
  mft-reco-sink
  mft-reco-merger
)

Set(Exe_Source
  src/runMFTrecoSampler.cxx
  src/runMFTrecoTaskProcessor.cxx
  src/runMFTrecoFileSink.cxx
  src/runMFTrecoMerger.cxx
)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

foreach (_file RANGE 0 ${_length}) # loop over a range because we traverse 2 lists and not 1
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
#  Set(EXE_NAME ${_name})
#  Set(SRCS ${_src})
#  Set(DEPENDENCIES CCDB)
  O2_GENERATE_EXECUTABLE(
      EXE_NAME ${_name}
      SOURCES ${_src}
      MODULE_LIBRARY_NAME ${LIBRARY_NAME}
      BUCKET_NAME ${BUCKET_NAME}
  )
endforeach (_file RANGE 0 ${_length})

#install(
#    FILES config/mft-dds.json
#    DESTINATION bin/config
#)
