# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(FV0Simulation
        SOURCES src/Detector.cxx
                src/Digitizer.cxx
                src/FV0DigParam.cxx
                src/Digits2Raw.cxx
        PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
                              O2::FV0Base
                              O2::DataFormatsFV0
                              O2::DetectorsRaw
                              ROOT::Physics
                              O2::Headers
                              O2::Framework)

o2_target_root_dictionary(FV0Simulation
        HEADERS include/FV0Simulation/Detector.h
                include/FV0Simulation/DigitizationConstant.h
                include/FV0Simulation/Digitizer.h
                include/FV0Simulation/FV0DigParam.h
                include/FV0Simulation/Digits2Raw.h)

o2_add_executable(digi2raw
        COMPONENT_NAME fv0
        SOURCES src/digit2raw.cxx
        PUBLIC_LINK_LIBRARIES O2::FV0Simulation
                              O2::DetectorsRaw
                              O2::DetectorsCommonDataFormats
                              O2::CommonUtils
                              Boost::program_options
                              O2::FV0Raw)

o2_data_file(COPY data  DESTINATION Detectors/FV0/simulation)

