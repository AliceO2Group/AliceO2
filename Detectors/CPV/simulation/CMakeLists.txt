# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(CPVSimulation
               SOURCES src/Detector.cxx
                       src/GeometryParams.cxx
                       src/Digitizer.cxx
                       src/RawWriter.cxx
               PUBLIC_LINK_LIBRARIES O2::DetectorsBase
                                     O2::DataFormatsCPV
                                     O2::CPVBase
                                     O2::CPVCalib
                                     O2::CCDB
                                     O2::SimConfig
                                     O2::SimulationDataFormat 
                                     O2::Headers 
                                     O2::DetectorsRaw)

o2_target_root_dictionary(CPVSimulation
                          HEADERS include/CPVSimulation/Detector.h
                                  include/CPVSimulation/GeometryParams.h
                                  include/CPVSimulation/Digitizer.h
                                  include/CPVSimulation/RawWriter.h)
o2_add_executable(digi2raw
                  COMPONENT_NAME cpv
                  PUBLIC_LINK_LIBRARIES O2::CPVSimulation
                  SOURCES src/RawCreator.cxx)

