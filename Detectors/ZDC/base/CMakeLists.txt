o2_add_library(ZDCBase
               SOURCES
               src/Geometry.cxx
               PUBLIC_LINK_LIBRARIES
               ROOT::MathCore
               FairRoot::Base)

o2_target_root_dictionary(ZDCBase HEADERS include/ZDCBase/Geometry.h)
