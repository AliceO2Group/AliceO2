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

o2_add_library(FOCALSimulation
    SOURCES src/Detector.cxx
    PUBLIC_LINK_LIBRARIES ROOT::TreePlayer O2::FOCALBase O2::DetectorsBase O2::SimConfig O2::SimulationDataFormat O2::Headers O2::DataFormatsFOCAL
    AliceO2::InfoLogger
    Microsoft.GSL::GSL)

o2_target_root_dictionary(
    FOCALSimulation
    HEADERS include/FOCALSimulation/Detector.h
)

o2_data_file(COPY geometryFiles  DESTINATION Detectors/Geometry/FOC/)
o2_data_file(COPY data DESTINATION Detectors/FOC/simulation)
