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

o2_add_test_root_macro(CreateCTPConfig.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(GetScalers.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(TestCTPScalers.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(SaveInputsConfig.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(CheckCTPDigits.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(CheckAOD2CTPDigits.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(CheckAOD2CTPDigitsII.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(TestConfig.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(dumpCTPRO.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(GetAndSave.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(ReadCTPRunScalersFromFile.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(GetScalersForRun.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(PlotPbLumi.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
o2_add_test_root_macro(PlotOrbit.C
                       PUBLIC_LINK_LIBRARIES O2::DataFormatsCTP
                                             O2::CCDB
                       LABELS ctp)
