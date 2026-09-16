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

# Compile-only: running it needs an o2sim geometry file, which does not exist at
# build time. When ACTS is missing this directory is not added at all, so the
# macro is exempted from the "every macro must be tested" check in
# cmake/O2RootMacroExclusionList.cmake.
o2_add_test_root_macro(CheckActsTrackingGeometry.C
                       PUBLIC_LINK_LIBRARIES O2::ACTSInterface
                                             O2::ALICE3ACTS
                                             O2::TRKBase
                       LABELS acts upgrades
                       COMPILE_ONLY)
