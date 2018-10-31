# DEPRECATED: Remove this file, once we require FairMQ 1.2+
#
# Simple check for availability of FairMQ
#
# The FairMQ module of FairRoot might be disabled in the built of FairRoot
# due to missing dependencies, e.g ZeroMQ and boost. Those dependencies
# also have to be available in the required (minimal) version.
#

if(FairRoot_DIR)
  set(FAIRROOTPATH ${FairRoot_DIR})
else()
  set(FAIRROOTPATH $ENV{FAIRROOTPATH})
endif(FairRoot_DIR)

if(FAIRROOTPATH)
  if(NOT FairMQInFairRoot_FIND_QUIETLY)
  MESSAGE(STATUS "FairRoot ... - found ${FAIRROOTPATH}")
  endif(NOT FairMQInFairRoot_FIND_QUIETLY)
else()
  if(NOT FairMQInFairRoot_FIND_QUIETLY)
  MESSAGE(FATAL_ERROR "FairRoot installation not found")
  endif(NOT FairMQInFairRoot_FIND_QUIETLY)
endif(FAIRROOTPATH)

set(FAIRMQ_REQUIRED_HEADERS FairMQDevice.h)
if(NOT FairMQInFairRoot_FIND_QUIETLY)
  message(STATUS "Looking for FairMQ functionality in FairRoot ...")
endif(NOT FairMQInFairRoot_FIND_QUIETLY)

find_path(FAIRMQ_INCLUDE_DIR NAMES ${FAIRMQ_REQUIRED_HEADERS}
  PATHS ${FAIRROOTPATH}/include/fairmq
  NO_DEFAULT_PATH
)

# search once more in the system if not yet found
find_path(FAIRMQ_INCLUDE_DIR NAMES ${FAIRMQ_REQUIRED_HEADERS}
)

if(FAIRMQ_INCLUDE_DIR)
  if(NOT FairMQInFairRoot_FIND_QUIETLY)
    message(STATUS "Looking for FairMQ functionality in FairRoot: yes")
  endif(NOT FairMQInFairRoot_FIND_QUIETLY)
  set(FAIRMQ_FOUND TRUE)
  set(FairMQInFairRoot_FOUND TRUE)
else(FAIRMQ_INCLUDE_DIR)
  if(FairMQInFairRoot_FIND_REQUIRED)
    message(FATAL_ERROR "FairRoot is not built with FairMQ support")
  else(FairMQInFairRoot_FIND_REQUIRED)
    if(NOT FairMQInFairRoot_FIND_QUIETLY)
      message(STATUS "Looking for FairMQ functionality in FairRoot: no")
    endif(NOT FairMQInFairRoot_FIND_QUIETLY)
  endif(FairMQInFairRoot_FIND_REQUIRED)
endif(FAIRMQ_INCLUDE_DIR)
