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

# TO DO: use the path from CMAKE and put it as an argument of alpideResponse.C (where it will be an argument of initData() !!!)

#add_subdirectory(Vbb-0.0V)
#add_subdirectory(Vbb-3.0V)

o2_add_test_root_macro(alpideResponse.C
    LABELS "its;mft")

#add_executable(alpide alpideResponse.C)
#add_custom_command(TARGET alpide
#    POST_BUILD
#    COMMAND root -b -q alpideResponse.C+
#    COMMENT "Storing AlpideResponseData for Vbb=0V and -3V in a .root file")

#set(FILE "hello.txt")

#add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
#    COMMAND ${CMAKE_COMMAND} echo "hello" > ${CMAKE_CURRENT_SOURCE_DIR}/${FILE})