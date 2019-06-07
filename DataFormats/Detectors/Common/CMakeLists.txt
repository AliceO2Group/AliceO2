o2_add_library(DetectorsCommonDataFormats
               SOURCES src/DetID.cxx src/AlignParam.cxx src/DetMatrixCache.cxx
               PUBLIC_LINK_LIBRARIES ROOT::Core ROOT::Geom MathUtils)

o2_target_root_dictionary(
  DetectorsCommonDataFormats
  HEADERS include/DetectorsCommonDataFormats/DetID.h
          include/DetectorsCommonDataFormats/AlignParam.h
          include/DetectorsCommonDataFormats/DetMatrixCache.h)

o2_add_test(DetID
            SOURCES test/testDetID.cxx
            PUBLIC_LINK_LIBRARIES DetectorsCommonDataFormats
            COMPONENT_NAME DetectorsCommonDataFormats)
