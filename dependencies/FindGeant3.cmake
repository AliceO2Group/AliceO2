# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# use the the config provided by the Geant3 installation but amend the target
# geant321 with the include directories

find_package(Geant3 NO_MODULE)
if(NOT Geant3_FOUND)
  return()
endif()

set_target_properties(geant321
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 "${Geant3_INCLUDE_DIRS}")
