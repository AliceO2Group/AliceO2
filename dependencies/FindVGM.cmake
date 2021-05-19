# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# use the VGMConfig.cmake provided by the VGM installation but amend the target VGM::XmlVGM with the include  and link
# directories

find_package(
  VGM
  NO_MODULE)
if(NOT
   VGM_FOUND)
  return()
endif()

# VGM uses namespace since 4.7
if(${VGM_VERSION}
   VERSION_LESS
   "4.7")
  set(targetVGM XmlVGM)
else()
  set(targetVGM VGM::XmlVGM)
endif()

set_target_properties(
  ${targetVGM}
  PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
             "${VGM_INCLUDE_DIRS}"
             INTERFACE_LINK_DIRECTORIES
             $<TARGET_FILE_DIR:${targetVGM}>)
