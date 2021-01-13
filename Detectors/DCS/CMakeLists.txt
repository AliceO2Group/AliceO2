# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# add_compile_options(-O0 -g -fPIC)

o2_add_library(
  DetectorsDCS
  TARGETVARNAME targetName
  SOURCES src/AliasExpander.cxx
          src/DCSProcessor.cxx
          src/DataPointCompositeObject.cxx
          src/DataPointCreator.cxx
          src/DataPointGenerator.cxx
          src/DataPointIdentifier.cxx
          src/DataPointValue.cxx
          src/DeliveryType.cxx
          src/GenericFunctions.cxx
          src/StringUtils.cxx
          src/Clock.cxx
  PUBLIC_LINK_LIBRARIES O2::Headers O2::CommonUtils O2::CCDB
                        O2::DetectorsCalibration ms_gsl::ms_gsl)

o2_target_root_dictionary(
  DetectorsDCS
  HEADERS include/DetectorsDCS/DataPointCompositeObject.h
          include/DetectorsDCS/DataPointIdentifier.h
          include/DetectorsDCS/DataPointValue.h
          include/DetectorsDCS/DCSProcessor.h)

if(OpenMP_CXX_FOUND)
  target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
  target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(BUILD_TESTING)
  o2_add_test(
    data-point-types
    SOURCES test/testDataPointTypes.cxx
    COMPONENT_NAME dcs
    LABELS "dcs"
    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsDCS)
  o2_add_test(
    alias-expander
    SOURCES test/testAliasExpander.cxx
    COMPONENT_NAME dcs
    LABELS "dcs"
    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsDCS)
  o2_add_test(
    data-point-generator
    SOURCES test/testDataPointGenerator.cxx
    COMPONENT_NAME dcs
    LABELS "dcs"
    PUBLIC_LINK_LIBRARIES O2::Framework O2::DetectorsDCS)
endif()

add_subdirectory(testWorkflow)
