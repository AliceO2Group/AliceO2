o2_add_library(T0Base
               SOURCES src/Geometry.cxx
               PUBLIC_LINK_LIBRARIES ROOT::Physics FairRoot::Base)

o2_target_root_dictionary(T0Base HEADERS include/T0Base/Geometry.h)

o2_file(COPY files DESTINATION share/Detectors/T0/)
