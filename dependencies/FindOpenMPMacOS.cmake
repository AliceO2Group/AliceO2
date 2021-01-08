find_library(OpenMP_LIBRARY
    NAMES omp
)

find_path(OpenMP_INCLUDE_DIR
    omp.h
)

mark_as_advanced(OpenMP_LIBRARY OpenMP_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenMP DEFAULT_MSG
    OpenMP_LIBRARY OpenMP_INCLUDE_DIR)

if (OpenMP_FOUND)
    set(OpenMP_LIBRARIES ${OpenMP_LIBRARY})
    set(OpenMP_INCLUDE_DIRS ${OpenMP_INCLUDE_DIR})
    set(OpenMP_COMPILE_OPTIONS -Xpreprocessor -fopenmp)

    set(OpenMP_CXX_FOUND True)
    set(OpenMPMacOS_FOUND True)
    add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
        IMPORTED_LOCATION ${OpenMP_LIBRARIES}
        INTERFACE_INCLUDE_DIRECTORIES "${OpenMP_INCLUDE_DIRS}"
        INTERFACE_COMPILE_OPTIONS "${OpenMP_COMPILE_OPTIONS}"
    )
endif()
