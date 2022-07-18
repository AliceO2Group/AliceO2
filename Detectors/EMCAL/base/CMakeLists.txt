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

o2_add_library(EMCALBase
        SOURCES src/Geometry.cxx src/Hit.cxx
        src/ShishKebabTrd1Module.cxx
        src/Mapper.cxx
        src/RCUTrailer.cxx
        src/ClusterFactory.cxx
        src/TriggerMappingV2.cxx
        PUBLIC_LINK_LIBRARIES O2::CommonDataFormat O2::Headers Boost::serialization
        O2::MathUtils O2::DataFormatsEMCAL
        O2::SimulationDataFormat ROOT::Physics)

o2_target_root_dictionary(EMCALBase
        HEADERS include/EMCALBase/GeometryBase.h
        include/EMCALBase/Geometry.h
        include/EMCALBase/Hit.h
        include/EMCALBase/ShishKebabTrd1Module.h
        include/EMCALBase/Mapper.h
        include/EMCALBase/RCUTrailer.h
        include/EMCALBase/TriggerMappingV2.h
        include/EMCALBase/ClusterFactory.h
)

o2_data_file(COPY files DESTINATION Detectors/EMC)

o2_add_test(Mapper
        SOURCES test/testMapper.cxx
        PUBLIC_LINK_LIBRARIES O2::EMCALBase
        COMPONENT_NAME emcal
        LABELS emcal
        ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test(RCUTrailer
        SOURCES test/testRCUTrailer.cxx
        PUBLIC_LINK_LIBRARIES O2::EMCALBase
        COMPONENT_NAME emcal
        LABELS emcal
        ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage)

o2_add_test_root_macro(test/testGeometryRowColIndexing.C
        PUBLIC_LINK_LIBRARIES O2::EMCALBase
        LABELS emcal)