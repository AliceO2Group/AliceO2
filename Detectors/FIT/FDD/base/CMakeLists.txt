o2_add_library(FDDBase
               SOURCES
               src/Geometry.cxx
               PUBLIC_LINK_LIBRARIES
               ROOT::Geom
               FairRoot::Base)

o2_target_root_dictionary(FDDBase HEADERS include/FDDBase/Geometry.h)
