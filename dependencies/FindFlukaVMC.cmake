# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# use the GEANT4_VMCConfig.cmake provided by the Geant4VMC installation but
# amend the target geant4vmc with the include directories

find_package(FlukaVMC NO_MODULE)
if(NOT FlukaVMC_FOUND)
return()
endif()

set_target_properties(flukavmc
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                "${FlukaVMC_INCLUDE_DIRS}"
                      		INTERFACE_LINK_DIRECTORIES
                                $<TARGET_FILE_DIR:flukavmc>)

# Promote the imported target to global visibility
# (so we can alias it)
set_target_properties(flukavmc PROPERTIES IMPORTED_GLOBAL TRUE)

add_library(MC::FlukaVMC ALIAS flukavmc)
