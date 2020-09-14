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

find_package(HepMC3 NO_MODULE)
if(NOT HepMC3_FOUND)
        return()
endif()

add_library(HepMC3 IMPORTED INTERFACE)

set_target_properties(HepMC3
        PROPERTIES
        INTERFACE_LINK_LIBRARIES "${HEPMC3_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${HEPMC3_INCLUDE_DIR}")

# Promote the imported target to global visibility (so we can alias it)
set_target_properties(HepMC3 PROPERTIES IMPORTED_GLOBAL TRUE)
add_library(MC::HepMC3 ALIAS HepMC3)
