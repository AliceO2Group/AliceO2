# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(PSRSimulation
               SOURCES src/PSRLayer.cxx
                       src/Detector.cxx
               PUBLIC_LINK_LIBRARIES O2::PSRBase
                                     O2::ITSMFTSimulation
                                     ROOT::Physics)

o2_target_root_dictionary(PSRSimulation
                          HEADERS include/PSRSimulation/Detector.h
                                  include/PSRSimulation/PSRLayer.h)

o2_data_file(COPY data  DESTINATION Detectors/PSR/simulation)
