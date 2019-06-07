set(ALIGPU_BUILD_TYPE "O2")

o2_add_header_only_library(CommonGPU
                           INTERFACE_LINK_LIBRARIES FairLogger::FairLogger)
