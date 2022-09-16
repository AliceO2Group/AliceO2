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
### Prepare the list of methods in FlagReasonFactory
file(READ "etc/flagReasons.csv" CSV_FLAG_REASONS)
# delete the CSV file header
string(REPLACE \"id\",\"method\",\"name\",\"bad\",\"obsolete\" "" CSV_FLAG_REASONS ${CSV_FLAG_REASONS})
# detects if there is obsolete flag '1' in the last column, adds [[deprecated]] if so and retains the rest of the string
string(REGEX REPLACE \([0-9]+,\".[^\"]*.\",.[^\"]*.,[0-1]\),1
        "[[deprecated]] \\1,1"
        CSV_FLAG_REASONS
        ${CSV_FLAG_REASONS})
# replaces the flag reason entry with a c++ method to create it
string(REGEX REPLACE \([0-9]+\),\"\(.[^\"]*.\)\",\(.[^\"]*.\),\([0-1]\),[0-1]
        "static FlagReason \\2\(\) { return { static_cast<uint16_t>\(\\1\), \\3, static_cast<bool>\(\\4\) }; }"
        CSV_FLAG_REASONS
        ${CSV_FLAG_REASONS})
# put the method lists inside the template
configure_file("include/DataFormatsQualityControl/FlagReasonFactory.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/DataFormatsQualityControl/FlagReasonFactory.h"
        @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/DataFormatsQualityControl/FlagReasonFactory.h"
        DESTINATION include/DataFormatsQualityControl)

o2_add_library(DataFormatsQualityControl
               SOURCES src/FlagReasons.cxx
                       src/TimeRangeFlag.cxx
                       src/TimeRangeFlagCollection.cxx
  PUBLIC_LINK_LIBRARIES O2::Headers
                        O2::FrameworkLogger
                        O2::DetectorsCommonDataFormats
  PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/include
                             include
        )

o2_target_root_dictionary(DataFormatsQualityControl
                          HEADERS include/DataFormatsQualityControl/FlagReasons.h
                                  include/DataFormatsQualityControl/TimeRangeFlag.h
                                  include/DataFormatsQualityControl/TimeRangeFlagCollection.h)



if(BUILD_TESTING)
o2_add_test(FlagReasons
  SOURCES test/testFlagReasons.cxx
  COMPONENT_NAME DataFormatsQualityControl
  PUBLIC_LINK_LIBRARIES O2::DataFormatsQualityControl
  TARGETVARNAME flagreasons)

target_include_directories(${flagreasons} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)

o2_add_test(TimeRangeFlag
            SOURCES test/testTimeRangeFlag.cxx
            COMPONENT_NAME DataFormatsQualityControl
            PUBLIC_LINK_LIBRARIES O2::DataFormatsQualityControl
            TARGETVARNAME timerangeflag)

target_include_directories(${timerangeflag} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

o2_add_test(TimeRangeFlagCollection
            SOURCES test/testTimeRangeFlagCollection.cxx
            COMPONENT_NAME DataFormatsQualityControl
            PUBLIC_LINK_LIBRARIES O2::DataFormatsQualityControl
            TARGETVARNAME timerangeflagcollection)

target_include_directories(${timerangeflagcollection} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)
endif()
