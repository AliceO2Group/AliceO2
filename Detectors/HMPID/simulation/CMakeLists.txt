# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(HMPIDSimulation
               SOURCES src/HMPIDDigitizer.cxx
                       src/Detector.cxx 
                       src/HmpidCoder2.cxx
               PUBLIC_LINK_LIBRARIES O2::HMPIDBase
                                     O2::SimulationDataFormat
                                     O2::DataFormatsHMP
                                     O2::DetectorsBase
                                     O2::DetectorsRaw
                                     ROOT::Physics)

o2_target_root_dictionary(HMPIDSimulation
                          HEADERS include/HMPIDSimulation/HmpidCoder2.h
                                  include/HMPIDSimulation/Detector.h
                                  include/HMPIDSimulation/HMPIDDigitizer.h)


o2_data_file(COPY data  DESTINATION Detectors/HMP/simulation)
