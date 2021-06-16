# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# Simply provide a namespaced alias for the existing target

set(_CMAKE_MODULE_PATH_COPY ${CMAKE_MODULE_PATH}) # Make sure CMAKE_MODULE_PATH is not reset
find_package(Arrow CONFIG QUIET)

if(NOT Arrow_FOUND)
        set(Arrow_DIR ${arrow_DIR})
        find_package(arrow CONFIG PATHS ${Arrow_DIR} QUIET)
endif()
find_package(Gandiva CONFIG PATHS ${Arrow_DIR} QUIET)

if(TARGET arrow_shared)
        # Promote the imported target to global visibility (so we can alias it)
        set_target_properties(arrow_shared PROPERTIES IMPORTED_GLOBAL TRUE)
        add_library(arrow::arrow_shared ALIAS arrow_shared)
        set(arrow_TARGET ON)
endif()

if(TARGET gandiva_shared)
        # Promote the imported target to global visibility (so we can alias it)
        set_target_properties(gandiva_shared PROPERTIES IMPORTED_GLOBAL TRUE)
        add_library(arrow::gandiva_shared ALIAS gandiva_shared)
        set(gandiva_TARGET ON)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(O2arrow REQUIRED_VARS arrow_TARGET gandiva_TARGET)

unset(arrow_TARGET)
unset(gandiva_TARGET)
set(CMAKE_MODULE_PATH ${_CMAKE_MODULE_PATH_COPY})
unset(_CMAKE_MODULE_PATH_COPY)
