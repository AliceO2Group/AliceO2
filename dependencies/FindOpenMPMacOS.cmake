find_library(OpenMP_LIBRARY
    NAMES omp libomp
    HINTS
        /opt/homebrew/opt/libomp/lib
        /opt/homebrew/lib
        /usr/local/opt/libomp/lib
        /usr/local/lib
)

find_path(OpenMP_INCLUDE_DIR
    NAMES omp.h
    HINTS
        /opt/homebrew/opt/libomp/include
        /opt/homebrew/include
        /usr/local/opt/libomp/include
        /usr/local/include
)

mark_as_advanced(OpenMP_LIBRARY OpenMP_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    OpenMPMacOS
    DEFAULT_MSG
    OpenMP_LIBRARY OpenMP_INCLUDE_DIR
)

if (OpenMPMacOS_FOUND)
    set(OpenMP_LIBRARIES ${OpenMP_LIBRARY})
    set(OpenMP_INCLUDE_DIRS ${OpenMP_INCLUDE_DIR})

    set(OpenMP_CXX_FOUND TRUE)
    set(OpenMP_FOUND TRUE)

    add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OpenMP_INCLUDE_DIRS}"
        INTERFACE_COMPILE_OPTIONS "-Xclang;-fopenmp"
        INTERFACE_LINK_LIBRARIES "${OpenMP_LIBRARIES}"
    )
    message(STATUS
        "Found OpenMP (macOS workaround): "
        "library=${OpenMP_LIBRARY}, "
        "include=${OpenMP_INCLUDE_DIR}"
    )
endif()
