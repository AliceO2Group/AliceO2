# Copyright 2019-2023 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

if (DEFINED ENV{O2_ENABLE_VTUNE})
  set(ENABLE_VTUNE_PROFILER TRUE)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(Vtune REQUIRED IMPORTED_TARGET ittnotify)

endif()
