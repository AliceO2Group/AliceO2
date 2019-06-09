set(ALIGPU_BUILD_TYPE "O2")

o2_add_library(GPUCommon
        PUBLIC_LINK_LIBRARIES FairLogger::FairLogger ROOT::RIO
        PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR})

o2_target_root_dictionary(GPUCommon
        HEADERS FlatObject.h
        LINKDEF GPUCommonLinkDef.h)
