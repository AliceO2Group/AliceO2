# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# Subdirectories will be compiled with O2 / AliRoot / Standalone To simplify the
# CMake, variables are defined for Sources / Headers first. Then, the actual
# CMake build scripts use these variables.
#
# SRCS: Common Sources for all builds HDRS_CINT: Headers for ROOT dictionary
# (always) HDRS_CINT_ALIROOT: Headers for ROOT dictionary (only in AliRoot)
# HDRS_CINT_O2: Headers for ROOT dictionary (only for O2) HDRS_INSTALL: Headers
# for installation only

add_subdirectory(Common)
add_subdirectory(TPCFastTransformation)
add_subdirectory(GPUTracking)
if(NOT ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_subdirectory(TPCSpaceChargeBase)
endif()
