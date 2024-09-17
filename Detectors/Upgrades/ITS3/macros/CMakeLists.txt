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

# Define custom ITS3 macro function for convenience
macro(its3_add_macro file)
  o2_add_test_root_macro(
    ${file}
    PUBLIC_LINK_LIBRARIES
      O2::ITSBase
      O2::ITS3Base
      O2::ITSMFTBase
      O2::ITSMFTSimulation
      O2::ITS3Reconstruction
      O2::ITS3Simulation
      O2::ITS3Align
      O2::MathUtils
      O2::SimulationDataFormat
      O2::DetectorsBase
      O2::Steer
    LABELS its3
    COMPILE_ONLY)
endmacro()

add_subdirectory(test)
add_subdirectory(eve)
add_subdirectory(align)
