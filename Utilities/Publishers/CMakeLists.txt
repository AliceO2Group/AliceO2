# FIXME: do we actually need a library here ?
o2_add_library(Publishers SOURCES src/DataPublisherDevice.cxx
        PUBLIC_LINK_LIBRARIES O2::Device)

o2_add_executable(datapublisher-device PUBLIC_LINK_LIBRARIES O2::Publishers
                  SOURCES src/runDataPublisherDevice.cxx)
