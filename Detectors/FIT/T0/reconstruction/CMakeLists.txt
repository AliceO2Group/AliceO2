o2_add_library(T0Reconstruction SOURCES src/CollisionTimeRecoTask.cxx
        PUBLIC_LINK_LIBRARIES O2::DataFormatsFITT0)

o2_target_root_dictionary(T0Reconstruction HEADERS
                          include/T0Reconstruction/CollisionTimeRecoTask.h)
