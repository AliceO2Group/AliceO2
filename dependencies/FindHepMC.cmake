# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# use the HepMCConfig.cmake provided by the HepMC3 installation to create a
# single target HepMC with the include directories and libraries we need

find_package(HepMC NO_MODULE)
if(NOT HepMC_FOUND)
  return()
endif()

add_library(HepMC IMPORTED INTERFACE)

set_target_properties(HepMC
                      PROPERTIES
		      INTERFACE_LINK_LIBRARIES "${HEPMC_LIBRARIES}"
		      INTERFACE_INCLUDE_DIRECTORIES "${HEPMC_INCLUDE_DIR}")

