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

find_package(VMC NO_MODULE)
if(NOT VMC_FOUND)
  return()
endif()

# Promote the imported target to global visibility
# (so we can alias it)
set_target_properties(VMCLibrary PROPERTIES IMPORTED_GLOBAL TRUE)
add_library(MC::VMC ALIAS VMCLibrary)
