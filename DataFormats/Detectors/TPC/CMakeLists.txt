# Comments
#
# * FIXME: ClusterNative depends on SimulationDataFormats : quite annoying for a
#   (supposedly real data component)
#
# * GPUCommonDef -> GPUCommonDef to follow more closely the rest of
#   the AliceO2 code, but probably the name should be simplified to remove some
#   common ...

o2_add_library(DataFormatsTPC
               SOURCES
               src/Helpers.cxx
               src/TrackTPC.cxx
               src/TPCSectorHeader.cxx
               src/ClusterNativeHelper.cxx
               PUBLIC_LINK_LIBRARIES
               O2::GPUCommon
               O2::SimulationDataFormat
               O2::Headers
               O2::Algorithm)

o2_target_root_dictionary(DataFormatsTPC
                          HEADERS
                          include/DataFormatsTPC/ClusterGroupAttribute.h
                          include/DataFormatsTPC/ClusterNative.h
                          include/DataFormatsTPC/ClusterNativeHelper.h
                          include/DataFormatsTPC/ClusterHardware.h
                          include/DataFormatsTPC/Helpers.h
                          include/DataFormatsTPC/TrackTPC.h
                          include/DataFormatsTPC/Constants.h
                          include/DataFormatsTPC/Defs.h
                          include/DataFormatsTPC/dEdxInfo.h
                          include/DataFormatsTPC/CompressedClusters.h)

o2_add_test(ClusterNative
            SOURCES
            test/testClusterNative.cxx
            COMPONENT_NAME
            DataFormats-TPC
            PUBLIC_LINK_LIBRARIES
            DataFormatsTPC)

o2_add_test(ClusterHardware
            SOURCES
            test/testClusterHardware.cxx
            COMPONENT_NAME
            DataFormats-TPC
            PUBLIC_LINK_LIBRARIES
            DataFormatsTPC)
