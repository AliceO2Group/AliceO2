o2_add_library(DataFormatsMID
               SOURCES
               src/ColumnData.cxx
               src/Track.cxx
               PUBLIC_LINK_LIBRARIES
               Boost::serialization
               O2::MathUtils)

o2_target_root_dictionary(DataFormatsMID
                          HEADERS
                          include/DataFormatsMID/Cluster2D.h
                          include/DataFormatsMID/Cluster3D.h
                          include/DataFormatsMID/ColumnData.h
                          include/DataFormatsMID/Track.h)
