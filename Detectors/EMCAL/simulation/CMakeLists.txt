# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(EMCALSimulation
               SOURCES src/Detector.cxx src/Digitizer.cxx src/DigitizerTask.cxx
                       src/SpaceFrame.cxx src/SimParam.cxx src/LabeledDigit.cxx
                       src/RawWriter.cxx src/DMAOutputStream.cxx
               PUBLIC_LINK_LIBRARIES O2::EMCALBase O2::DetectorsBase O2::SimConfig O2::SimulationDataFormat O2::Headers)

o2_target_root_dictionary(EMCALSimulation
                          HEADERS include/EMCALSimulation/Detector.h
                                  include/EMCALSimulation/Digitizer.h
                                  include/EMCALSimulation/DigitizerTask.h
                                  include/EMCALSimulation/RawWriter.h
                                  include/EMCALSimulation/DMAOutputStream.h
                                  include/EMCALSimulation/SpaceFrame.h
                                  include/EMCALSimulation/SimParam.h
                                  include/EMCALSimulation/LabeledDigit.h)

o2_data_file(COPY data DESTINATION Detectors/EMC/simulation)
