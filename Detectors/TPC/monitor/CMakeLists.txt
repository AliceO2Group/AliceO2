o2_add_library(TPCMonitor
               SOURCES
               src/SimpleEventDisplay.cxx
               PUBLIC_LINK_LIBRARIES
               O2::TPCBase
               O2::TPCCalibration)

o2_target_root_dictionary(TPCMonitor HEADERS
                          include/TPCMonitor/SimpleEventDisplay.h)

o2_add_executable(monitor
                  COMPONENT_NAME
                  tpc
                  SOURCES
                  run/runMonitor.cxx
                  PUBLIC_LINK_LIBRARIES
                  O2::TPCMonitor)

o2_add_test_root_macro(macro/RunCompareMode3.C PUBLIC_LINK_LIBRARIES
                       O2::TPCReconstruction LABELS tpc)
o2_add_test_root_macro(macro/RunFindAdcError.C PUBLIC_LINK_LIBRARIES
                       O2::TPCReconstruction LABELS tpc)
o2_add_test_root_macro(macro/RunSimpleEventDisplay.C PUBLIC_LINK_LIBRARIES
                       O2::TPCMonitor LABELS tpc)
