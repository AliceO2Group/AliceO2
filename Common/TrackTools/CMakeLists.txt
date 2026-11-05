# Copyright 2026-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# Expose and install the shared public TrackTools headers without compiled sources.
o2_add_header_only_library(
  TrackTools
  INTERFACE_LINK_LIBRARIES
    O2::CommonConstants
    O2::MathUtils
    O2::ReconstructionDataFormats
    O2::TPCBase
    O2::DetectorsBase)

# Keep the portable PyROOT/pytest validation available from the installed O2 module.
set(TRACKTOOLS_TEST_INSTALL_DIR share/test/TrackTools)
install(FILES
  test/python/conftest.py
  test/python/test_full_state_radius.py
  test/python/test_sector_rows_primary_baseline.py
  DESTINATION ${TRACKTOOLS_TEST_INSTALL_DIR}/python)
install(FILES test/README_TESTS.md
  DESTINATION ${TRACKTOOLS_TEST_INSTALL_DIR})
