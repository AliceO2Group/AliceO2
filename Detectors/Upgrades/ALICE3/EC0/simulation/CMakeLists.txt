# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(EC0Simulation
               SOURCES src/EC0Layer.cxx
                       src/Detector.cxx
               PUBLIC_LINK_LIBRARIES O2::EC0Base
                                     O2::ITSMFTSimulation
                                     ROOT::Physics)

o2_target_root_dictionary(EC0Simulation
                          HEADERS include/EC0Simulation/Detector.h
                                  include/EC0Simulation/EC0Layer.h)

o2_data_file(COPY data  DESTINATION Detectors/EC0/simulation)
