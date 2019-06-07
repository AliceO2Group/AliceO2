# TODO: remove this file once FairRoot correctly exports its cmake config

find_path(FairRoot_INC FairDetector.h ${FairRoot_ROOT})

if(NOT EXISTS ${FairRoot_INC})
  return()
endif()

get_filename_component(FairRoot_DIR "${FairRoot_INC}/.." ABSOLUTE)

find_library(FairRoot_Tools FairTools ${FairRoot_ROOT})
find_library(FairRoot_ParBase ParBase ${FairRoot_ROOT})
find_library(FairRoot_GeoBase GeoBase ${FairRoot_ROOT})
find_library(FairRoot_Base Base ${FairRoot_ROOT})
find_library(FairRoot_ParMQ ParMQ ${FairRoot_ROOT})
find_library(FairRoot_Gen Gen ${FairRoot_ROOT})

if(NOT TARGET FairRoot::Tools)
  add_library(FairRoot::Tools IMPORTED INTERFACE)
  set_target_properties(FairRoot::Tools
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${FairRoot_INC}
                                   INTERFACE_LINK_LIBRARIES ${FairRoot_Tools})
  target_link_libraries(FairRoot::Tools INTERFACE FairLogger::FairLogger)
endif()

if(NOT TARGET FairRoot::ParBase)
  add_library(FairRoot::ParBase IMPORTED INTERFACE)
  set_target_properties(FairRoot::ParBase
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${FairRoot_INC}
                                   INTERFACE_LINK_LIBRARIES ${FairRoot_ParBase})
  target_link_libraries(FairRoot::ParBase INTERFACE FairRoot::Tools)
endif()

if(NOT TARGET FairRoot::GeoBase)
  add_library(FairRoot::GeoBase IMPORTED INTERFACE)
  set_target_properties(FairRoot::GeoBase
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${FairRoot_INC}
                                   INTERFACE_LINK_LIBRARIES ${FairRoot_GeoBase})
endif()

if(NOT TARGET FairRoot::Base)
  add_library(FairRoot::Base IMPORTED INTERFACE)
  set_target_properties(FairRoot::Base
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${FairRoot_INC}
                                   INTERFACE_LINK_LIBRARIES ${FairRoot_Base})
  target_link_libraries(FairRoot::Base
                        INTERFACE FairRoot::Tools FairRoot::ParBase
                                  FairRoot::GeoBase ROOT::ROOTDataFrame)
  if(TARGET arrow_shared)
    # FIXME: this dependency (coming from ROOTDataFrame) should be handled in
    # ROOT itself
    target_link_libraries(FairRoot::Base INTERFACE arrow_shared)
  endif()
endif()

if(NOT TARGET FairRoot::ParMQ)
  add_library(FairRoot::ParMQ IMPORTED INTERFACE)
  set_target_properties(FairRoot::ParMQ
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${FairRoot_INC}
                                   INTERFACE_LINK_LIBRARIES ${FairRoot_ParMQ})
  target_link_libraries(FairRoot::ParMQ
                        INTERFACE FairRoot::ParBase FairMQ::FairMQ)
  if(TARGET arrow_shared)
    # FIXME: this dependency (coming from ROOTDataFrame) should be handled in
    # ROOT itself
    target_link_libraries(FairRoot::ParMQ INTERFACE arrow_shared)
  endif()
endif()

if(NOT TARGET FairRoot::Gen)
  add_library(FairRoot::Gen IMPORTED INTERFACE)
  set_target_properties(FairRoot::Gen
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${FairRoot_INC}
                                   INTERFACE_LINK_LIBRARIES ${FairRoot_Gen})
  target_link_libraries(FairRoot::Gen
                        INTERFACE FairRoot::ParBase FairRoot::Base
                                  FairRoot::ParBase)
endif()
