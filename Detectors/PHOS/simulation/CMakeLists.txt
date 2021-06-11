#Copyright CERN and copyright holders of ALICE O2.This software is distributed
#under the terms of the GNU General Public License v3(GPL Version 3), copied
#verbatim in the file "COPYING".
#
#See http: //alice-o2.web.cern.ch/license for full licensing information.
#
#In applying this license CERN does not waive the privileges and immunities
#granted to it by virtue of its status as an Intergovernmental Organization or
#submit itself to any jurisdiction.

o2_add_library(PHOSSimulation
               SOURCES src/Detector.cxx
                       src/GeometryParams.cxx
                       src/Digitizer.cxx
                       src/RawWriter.cxx
               PUBLIC_LINK_LIBRARIES O2::DetectorsBase
                                     O2::DataFormatsPHOS
                                     O2::PHOSBase
                                     O2::SimConfig
                                     O2::SimulationDataFormat
                                     O2::Headers
                                     O2::DetectorsRaw)

o2_target_root_dictionary(PHOSSimulation
                          HEADERS include/PHOSSimulation/Detector.h
                                  include/PHOSSimulation/GeometryParams.h
                                  include/PHOSSimulation/Digitizer.h
                                  include/PHOSSimulation/RawWriter.h)
o2_add_executable(digi2raw
                  COMPONENT_NAME phos
                  PUBLIC_LINK_LIBRARIES O2::PHOSSimulation
                  SOURCES src/RawCreator.cxx)


o2_data_file(COPY data  DESTINATION Detectors/PHS/simulation)
