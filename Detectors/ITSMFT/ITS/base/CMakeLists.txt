o2_add_library(ITSBase
               SOURCES
               src/GeometryTGeo.cxx
               src/ContainerFactory.cxx
               src/MisalignmentParameter.cxx
               PUBLIC_LINK_LIBRARIES
               O2::DetectorsBase
               O2::ITSMFTBase)

o2_target_root_dictionary(ITSBase HEADERS include/ITSBase/GeometryTGeo.h
                          include/ITSBase/ContainerFactory.h
                          include/ITSBase/MisalignmentParameter.h)
