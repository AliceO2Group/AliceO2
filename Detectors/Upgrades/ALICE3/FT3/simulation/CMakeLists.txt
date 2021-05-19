# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(
  FT3Simulation
  SOURCES src/FT3Layer.cxx
          src/Detector.cxx
  PUBLIC_LINK_LIBRARIES
    O2::FT3Base
    O2::ITSMFTSimulation
    ROOT::Physics)

o2_target_root_dictionary(
  FT3Simulation
  HEADERS include/FT3Simulation/Detector.h
          include/FT3Simulation/FT3Layer.h)

o2_data_file(COPY data DESTINATION Detectors/FT3/simulation)
