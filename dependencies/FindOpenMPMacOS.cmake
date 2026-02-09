# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

find_library(OpenMP_LIBRARY
    NAMES omp libomp
    HINTS
        /opt/homebrew/opt/libomp/lib
        /usr/local/opt/libomp/lib
)

find_path(OpenMP_INCLUDE_DIR
    NAMES omp.h
    HINTS
        /opt/homebrew/opt/libomp/include
        /usr/local/opt/libomp/include
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
