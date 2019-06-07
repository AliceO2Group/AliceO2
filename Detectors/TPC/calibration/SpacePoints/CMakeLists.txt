o2_add_library(SpacePoints
               SOURCES
               src/Param.cxx
               src/TrackResiduals.cxx
               PUBLIC_LINK_LIBRARIES
               O2::DataFormatsTPC)

o2_target_root_dictionary(SpacePoints
                          HEADERS
                          include/SpacePoints/Param.h
                          include/SpacePoints/TrackResiduals.h
                          LINKDEF
                          src/SpacePointCalibLinkDef.h)
