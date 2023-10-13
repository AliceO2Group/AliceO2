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

# setup files to be installed (only ROOT macros for the moment) NOT using GLOB,
# as we should be mindful of what we install. if we have lot of files here, it's
# probably because most of them should be elsewhere in the first place ...

install(FILES extractCTF.C
              dumpCTF.C
              CTFdict2CCDBfiles.C
        DESTINATION share/macro/)

o2_add_test_root_macro(extractCTF.C
                       PUBLIC_LINK_LIBRARIES O2::CTFWorkflow
                       LABELS ctf)

o2_add_test_root_macro(dumpCTF.C
                       PUBLIC_LINK_LIBRARIES O2::CTFWorkflow
                       LABELS ctf)

o2_add_test_root_macro(CTFdict2CCDBfiles.C
                       PUBLIC_LINK_LIBRARIES O2::CTFWorkflow fmt::fmt
                       LABELS ctf COMPILE_ONLY)

o2_add_test_root_macro(convCTFDict.C
                       PUBLIC_LINK_LIBRARIES O2::CTFWorkflow fmt::fmt
                       LABELS ctf COMPILE_ONLY)
