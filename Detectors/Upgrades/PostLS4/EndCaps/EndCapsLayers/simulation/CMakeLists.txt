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
               SOURCES src/V11Geometry.cxx src/V1Layer.cxx src/V3Layer.cxx
                       src/Detector.cxx src/V3Services.cxx
               PUBLIC_LINK_LIBRARIES O2::ECLayersBase O2::EndCapsSimulation
                                     ROOT::Physics)

o2_target_root_dictionary(EC0Simulation
                          HEADERS include/EC0Simulation/Detector.h
                                  include/EC0Simulation/V1Layer.h
                                  include/EC0Simulation/V3Layer.h
                                  include/EC0Simulation/V11Geometry.h
                                  include/EC0Simulation/V3Services.h)

o2_data_file(COPY data  DESTINATION Detectors/EC0/simulation)

o2_add_executable(digi2raw
                  COMPONENT_NAME ec0
                  SOURCES src/digi2raw.cxx
                  PUBLIC_LINK_LIBRARIES O2::EndCapsReconstruction
                                        O2::DataFormatsITSMFT
                                        O2::EndCapsBase
                                        O2::EndCapsSimulation
                                        O2::DetectorsRaw
                                        O2::DetectorsCommonDataFormats
                                        O2::CommonUtils
                                        Boost::program_options)
