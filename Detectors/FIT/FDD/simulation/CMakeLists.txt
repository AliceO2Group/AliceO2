# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(FDDSimulation
               SOURCES src/Detector.cxx
                       src/Digitizer.cxx
                       src/Digits2Raw.cxx
               PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
                                     O2::FDDBase
                                     O2::DataFormatsFDD
                                     O2::DetectorsRaw
                                     ROOT::Physics)

o2_target_root_dictionary(FDDSimulation
                          HEADERS include/FDDSimulation/Detector.h
                                  include/FDDSimulation/Digitizer.h
                                  include/FDDSimulation/DigitizationParameters.h
                                  include/FDDSimulation/Digits2Raw.h)

o2_add_executable(digit2raw
        COMPONENT_NAME fdd
        SOURCES src/digit2raw.cxx
        PUBLIC_LINK_LIBRARIES O2::FDDSimulation
                              O2::DetectorsRaw
                              O2::DetectorsCommonDataFormats
                              O2::CommonUtils
                              Boost::program_options)