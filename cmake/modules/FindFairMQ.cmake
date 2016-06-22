#
# Simple check for availability of FairMQ
#
# The FairMQ module of FairRoot might be disabled in the built of FairRoot
# due to missing dependencies, e.g ZeroMQ and boost. Those dependencies
# also have to be available in the required (minimal) version.
#

set(FAIRMQ_REQUIRED_HEADERS FairMQDevice.h)
if(NOT FairMQ_FIND_QUIETLY)
  message(STATUS "Looking for FairMQ functionality in FairRoot ...")
endif(NOT FairMQ_FIND_QUIETLY)

find_path(FAIRMQ_INCLUDE_DIR NAMES ${FAIRMQ_REQUIRED_HEADERS}
  PATHS ${FairRoot_DIR}/include
  NO_DEFAULT_PATH
)

# search once more in the system if not yet found
find_path(FAIRMQ_INCLUDE_DIR NAMES ${FAIRMQ_REQUIRED_HEADERS}
)

if(FAIRMQ_INCLUDE_DIR)
  if(NOT FairMQ_FIND_QUIETLY)
    message(STATUS "Looking for FairMQ functionality in FairRoot: yes")
  endif(NOT FairMQ_FIND_QUIETLY)
  set(FAIRMQ_FOUND TRUE)
else(FAIRMQ_INCLUDE_DIR)
  if(FairMQ_FIND_REQUIRED)
    message(FATAL_ERROR "FairRoot is not built with FairMQ support")
  else(FairMQ_FIND_REQUIRED)
    if(NOT FairMQ_FIND_QUIETLY)
      message(STATUS "Looking for FairMQ functionality in FairRoot: no")
    endif(NOT FairMQ_FIND_QUIETLY)
  endif(FairMQ_FIND_REQUIRED)
endif(FAIRMQ_INCLUDE_DIR)
