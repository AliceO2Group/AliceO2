add_subdirectory(SpacePoints)

o2_add_library(TPCCalibration
               SOURCES
               src/CalibRawBase.cxx
               src/CalibPedestal.cxx
               src/CalibPulser.cxx
               src/CalibTreeDump.cxx
               PUBLIC_LINK_LIBRARIES
               O2::DataFormatsTPC
               O2::TPCBase
               O2::TPCReconstruction
               ROOT::Minuit)

o2_target_root_dictionary(TPCCalibration
                          HEADERS
                          include/TPCCalibration/CalibRawBase.h
                          include/TPCCalibration/CalibPedestal.h
                          include/TPCCalibration/CalibPulser.h
                          include/TPCCalibration/CalibTreeDump.h)

o2_add_test_root_macro(macro/comparePedestalsAndNoise.C PUBLIC_LINK_LIBRARIES
                       TPCBase)
o2_add_test_root_macro(macro/drawNoiseAndPedestal.C PUBLIC_LINK_LIBRARIES
                       TPCBase)
o2_add_test_root_macro(macro/runPedestal.C PUBLIC_LINK_LIBRARIES TPCCalibration)
