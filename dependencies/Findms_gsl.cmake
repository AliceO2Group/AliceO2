find_path(MS_GSL_INCLUDE_DIR gsl/gsl
          PATHS ${ms_gsl_ROOT}/include
          NO_DEFAULT_PATH)
if(NOT MS_GSL_INCLUDE_DIR)
  set(MS_GSL_FOUND FALSE)
  message(FATAL_ERROR "MS_GSL not found")
  return()
endif()

set(MS_GSL_FOUND TRUE)

if(NOT TARGET ms_gsl::ms_gsl)
  add_library(ms_gsl::ms_gsl INTERFACE IMPORTED)
  set_target_properties(ms_gsl::ms_gsl
                        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                   ${MS_GSL_INCLUDE_DIR})
endif()

mark_as_advanced(MS_GSL_INCLUDE_DIR)
