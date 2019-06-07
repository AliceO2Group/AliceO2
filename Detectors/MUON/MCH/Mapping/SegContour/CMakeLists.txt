o2_add_library(MCHMappingSegContour3 SOURCES src/CathodeSegmentationContours.cxx
               src/CathodeSegmentationSVGWriter.cxx
               src/SegmentationContours.cxx
               PUBLIC_LINK_LIBRARIES O2::MCHMappingImpl3 O2::MCHContour
               Boost::boost)

o2_add_executable(mapping-svg-segmentation3
                  COMPONENT_NAME
                  mch
                  SOURCES
                  src/SVGSegmentation.cxx
                  PUBLIC_LINK_LIBRARIES
                  O2::MCHMappingSegContour3
                  Boost::program_options)
