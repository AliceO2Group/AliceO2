include(FindPackageHandleStandardArgs)
find_path(ALICE_GRID_UTILS_INCLUDE_DIR "TAliceCollection.h"
        PATH_SUFFIXES "include"
        HINTS "${ALICE_GRID_UTILS_ROOT}")
#get_filename_component(ALICE_GRID_UTILS_INCLUDE_DIR ${ALICE_GRID_UTILS_INCLUDE_DIR} DIRECTORY)
find_package_handle_standard_args(AliceGridUtils DEFAULT_MSG ALICE_GRID_UTILS_INCLUDE_DIR)
include_directories(${ALICE_GRID_UTILS_INCLUDE_DIR})
