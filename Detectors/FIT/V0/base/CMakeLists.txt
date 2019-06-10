o2_add_library(V0Base SOURCES src/Geometry.cxx
        PUBLIC_LINK_LIBRARIES ROOT::Geom
        FairRoot::Base)

o2_target_root_dictionary(V0Base HEADERS include/V0Base/Geometry.h)

o2_data_file(COPY files DESTINATION Detectors/V0/)
