# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# Simply provide a namespaced alias for the existing target

find_package(Arrow CONFIG QUIET)
if(NOT Arrow_FOUND)
        find_package(arrow CONFIG QUIET)
        set(Arrow_DIR ${arrow_DIR})
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
