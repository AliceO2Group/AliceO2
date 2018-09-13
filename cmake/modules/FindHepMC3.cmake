# **************************************************************************
# * Copyright(c) 1998-2015, ALICE Experiment at CERN, All rights reserved. *
# *                                                                        *
# * Author: The ALICE Off-line Project.                                    *
# * Contributors are mentioned in the code where appropriate.              *
# *                                                                        *
# * Permission to use, copy, modify and distribute this software and its   *
# * documentation strictly for non-commercial purposes is hereby granted   *
# * without fee, provided that the above copyright notice appears in all   *
# * copies and that both the copyright notice and this permission notice   *
# * appear in the supporting documentation. The authors make no claims     *
# * about the suitability of this software for any purpose. It is          *
# * provided "as is" without express or implied warranty.                  *
# **************************************************************************

set(HepMC3_FOUND FALSE)
message(STATUS "Looking for HepMC...")

if(HEPMC3_DIR)

  if(EXISTS ${HEPMC3_DIR}/share/HepMC/cmake/HepMCConfig.cmake)
    include(${HEPMC3_DIR}/share/HepMC/cmake/HepMCConfig.cmake)
    set(HepMC3_FOUND TRUE)
    message(STATUS "Looking for HepMC3... - found ${HEPMC3_DIR}")

  else()

    message(STATUS "Looking for HepMC3... - not found")

  endif()

endif()

if(NOT HepMC3_FOUND)
  if(HepMC3_FIND_REQUIRED)
    message(FATAL_ERROR "Please point to the HepMC3 installation using -DHEPMC3_DIR=<HEPMC3_INSTALL_DIR>")
  endif()
endif()
