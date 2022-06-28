#  Copyright 2019-2020 CERN and copyright holders of ALICE O2.
#  See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
#  All rights not expressly granted are reserved.

#  This software is distributed under the terms of the GNU General Public
#  License v3 (GPL Version 3), copied verbatim in the file "COPYING".

#  In applying this license CERN does not waive the privileges and immunities
#  granted to it by virtue of its status as an Intergovernmental Organization
#  or submit itself to any jurisdiction.

#  @author SwirtaB


o2_add_library(FASTSimulation
        SOURCES src/FastSimulations.cxx src/Processors.cxx src/Utils.cxx
        PUBLIC_LINK_LIBRARIES ONNXRuntime::ONNXRuntime)

o2_data_file(COPY scales DESTINATION Detectors/ZDC/fastsimulation)

o2_add_executable(fastsim-example
        SOURCES tests/run-example.cxx
        PUBLIC_LINK_LIBRARIES O2::FASTSimulation
        COMPONENT_NAME zdc)

