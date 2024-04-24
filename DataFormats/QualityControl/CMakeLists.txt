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

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)
### Prepare the list of methods in FlagTypeFactory
file(READ "etc/flagTypes.csv" CSV_FLAG_TYPES)
# delete the CSV file header
string(REPLACE \"id\",\"method\",\"name\",\"bad\",\"obsolete\" "" CSV_FLAG_TYPES ${CSV_FLAG_TYPES})
# detects if there is obsolete flag '1' in the last column, adds [[deprecated]] if so and retains the rest of the string
string(REGEX REPLACE \([0-9]+,\".[^\"]*.\",.[^\"]*.,[0-1]\),1
        "[[deprecated]] \\1,1"
        CSV_FLAG_TYPES
        ${CSV_FLAG_TYPES})
# replaces the flag reason entry with a c++ method to create it
string(REGEX REPLACE \([0-9]+\),\"\(.[^\"]*.\)\",\(.[^\"]*.\),\([0-1]\),[0-1]
        "static FlagType \\2\(\) { return { static_cast<uint16_t>\(\\1\), \\3, static_cast<bool>\(\\4\) }; }"
        CSV_FLAG_TYPES
        ${CSV_FLAG_TYPES})
# put the method lists inside the template
configure_file("include/DataFormatsQualityControl/FlagTypeFactory.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/DataFormatsQualityControl/FlagTypeFactory.h"
        @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/DataFormatsQualityControl/FlagTypeFactory.h"
        DESTINATION include/DataFormatsQualityControl)

o2_add_library(DataFormatsQualityControl
               SOURCES src/FlagType.cxx
                       src/QualityControlFlag.cxx
                       src/QualityControlFlagCollection.cxx
  PUBLIC_LINK_LIBRARIES O2::FrameworkLogger
                        O2::DetectorsCommonDataFormats
  PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/include
                             include
        )

o2_target_root_dictionary(DataFormatsQualityControl
                          HEADERS include/DataFormatsQualityControl/FlagType.h
                                  include/DataFormatsQualityControl/QualityControlFlag.h
                                  include/DataFormatsQualityControl/QualityControlFlagCollection.h)


if(BUILD_TESTING)
o2_add_test(FlagTypes
  SOURCES test/testFlagTypes.cxx
  COMPONENT_NAME DataFormatsQualityControl
  PUBLIC_LINK_LIBRARIES O2::DataFormatsQualityControl
  TARGETVARNAME flagtypes)

target_include_directories(${flagtypes} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)

o2_add_test(QualityControlFlag
            SOURCES test/testQualityControlFlag.cxx
            COMPONENT_NAME DataFormatsQualityControl
            PUBLIC_LINK_LIBRARIES O2::DataFormatsQualityControl
            TARGETVARNAME qualitycontrolflag)

target_include_directories(${qualitycontrolflag} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

o2_add_test(QualityControlFlagCollection
            SOURCES test/testQualityControlFlagCollection.cxx
            COMPONENT_NAME DataFormatsQualityControl
            PUBLIC_LINK_LIBRARIES O2::DataFormatsQualityControl
            TARGETVARNAME qualitycontrolflagcollection)

target_include_directories(${qualitycontrolflagcollection} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)
endif()
