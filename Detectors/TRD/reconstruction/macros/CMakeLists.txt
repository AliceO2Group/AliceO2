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

o2_add_test_root_macro(checkTrackletCharges.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsTRD
                       LABELS trd COMPILE_ONLY)

o2_add_test_root_macro(CompareDigitsAndTracklets.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsTRD
                       LABELS trd COMPILE_ONLY)

o2_add_test_root_macro(createLinkToHCIDMapping.C
                      PUBLIC_LINK_LIBRARIES O2::DataFormatsTRD
                                            O2::CCDB
                      LABELS trd COMPILE_ONLY)

o2_add_test_root_macro(checkRawStats.C
                      PUBLIC_LINK_LIBRARIES O2::DataFormatsTRD
                      LABELS trd COMPILE_ONLY)


install(
  FILES checkTrackletCharges.C
  CompareDigitsAndTracklets.C
  createLinkToHCIDMapping.C
  checkRawStats.C
  DESTINATION share/macro/)
