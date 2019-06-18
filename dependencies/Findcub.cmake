
find_path(CUB_INCLUDE_DIR cub/cub.cuh
          PATHS ${cub_ROOT}
          NO_DEFAULT_PATH)

if(NOT CUB_INCLUDE_DIR)
  set(CUB_FOUND FALSE)
  message(FATAL_ERROR "CUB not found")
  return()
endif()

set(CUB_FOUND TRUE)

if(NOT TARGET cub::cub)
  add_library(cub::cub INTERFACE IMPORTED)
  set_target_properties(cub::cub
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                   ${CUB_INCLUDE_DIR})
endif()

mark_as_advanced(CUB_INCLUDE_DIR)

