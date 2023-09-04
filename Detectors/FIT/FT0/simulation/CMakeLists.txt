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

o2_add_library(FT0Simulation
        SOURCES src/Detector.cxx
                src/Digitizer.cxx
        PUBLIC_LINK_LIBRARIES O2::SimulationDataFormat
                              O2::Framework
                              O2::DetectorsRaw
                              O2::Headers
                              O2::FT0Base
                              O2::DataFormatsFT0
                              O2::CCDB
                              O2::DetectorsCalibration
                              )

o2_target_root_dictionary(FT0Simulation
        HEADERS include/FT0Simulation/Detector.h
                include/FT0Simulation/DigitizationConstants.h
                include/FT0Simulation/Digitizer.h
                )

o2_add_executable(digi2raw
        COMPONENT_NAME ft0
        SOURCES src/digit2raw.cxx
        PUBLIC_LINK_LIBRARIES O2::FT0Raw
                              Boost::program_options)

o2_data_file(COPY data DESTINATION Detectors/FT0/simulation)

