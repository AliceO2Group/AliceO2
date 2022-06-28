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

o2_add_library(EMCALSimulation
               SOURCES src/Detector.cxx src/Digitizer.cxx src/SDigitizer.cxx
                       src/DigitsWriteoutBuffer.cxx src/DigitsVectorStream.cxx src/SpaceFrame.cxx src/SimParam.cxx
                       src/LabeledDigit.cxx src/RawWriter.cxx
               PUBLIC_LINK_LIBRARIES O2::EMCALBase O2::DetectorsBase O2::SimConfig O2::SimulationDataFormat O2::Headers O2::DetectorsRaw O2::EMCALReconstruction O2::DataFormatsCTP)

o2_target_root_dictionary(EMCALSimulation
                          HEADERS include/EMCALSimulation/Detector.h
                                  include/EMCALSimulation/Digitizer.h
                                  include/EMCALSimulation/SDigitizer.h
                                  include/EMCALSimulation/DigitsWriteoutBuffer.h
                                  include/EMCALSimulation/DigitsVectorStream.h
                                  include/EMCALSimulation/RawWriter.h
                                  include/EMCALSimulation/SpaceFrame.h
                                  include/EMCALSimulation/SimParam.h
                                  include/EMCALSimulation/LabeledDigit.h)

o2_add_executable(rawcreator
                  COMPONENT_NAME emcal
                  PUBLIC_LINK_LIBRARIES O2::EMCALSimulation
                  SOURCES src/RawCreator.cxx)

o2_data_file(COPY data DESTINATION Detectors/EMC/simulation)
