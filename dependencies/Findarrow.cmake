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

find_package(arrow CONFIG)
if(NOT arrow_FOUND)
  return()
endif()

# Promote the imported target to global visibility (so we can alias it)
set_target_properties(arrow_shared PROPERTIES IMPORTED_GLOBAL TRUE)

add_library(arrow::arrow_shared ALIAS arrow_shared)
