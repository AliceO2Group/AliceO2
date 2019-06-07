o2_add_library(TOFReconstruction SOURCES src/DataReader.cxx src/Clusterer.cxx
               src/ClustererTask.cxx
               PUBLIC_LINK_LIBRARIES O2::TOFBase
               O2::DataFormatsTOF
               O2::SimulationDataFormat)

o2_target_root_dictionary(TOFReconstruction HEADERS
                          include/TOFReconstruction/DataReader.h
                          include/TOFReconstruction/Clusterer.h
                          include/TOFReconstruction/ClustererTask.h)
