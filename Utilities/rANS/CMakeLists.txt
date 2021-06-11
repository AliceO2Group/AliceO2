# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

o2_add_library(rANS  
               SOURCES src/SymbolStatistics.cxx
                       src/FrequencyTable.cxx
               PUBLIC_LINK_LIBRARIES FairLogger::FairLogger)


o2_add_test(EncodeDecode
            NAME EncodeDecode
            SOURCES test/test_ransEncodeDecode.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            LABELS utils)
            
o2_add_test(FrequencyTable
            NAME FrequencyTable
            SOURCES test/test_ransFrequencyTable.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            LABELS utils)
            
o2_add_test(CombinedIterator
            NAME CombinedIterator
            SOURCES test/test_ransCombinedIterator.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            LABELS utils)
            
if (TARGET benchmark::benchmark)
o2_add_executable(CombinedIterator
                    SOURCES benchmarks/bench_ransCombinedIterator.cxx
                    COMPONENT_NAME rANS
              IS_BENCHMARK
                    PUBLIC_LINK_LIBRARIES O2::rANS benchmark::benchmark)
endif()
            
o2_add_executable(rans-encode-decode-8
                  TARGETVARNAME targetName
                  SOURCES run/bin-encode-decode.cxx
                  PUBLIC_LINK_LIBRARIES O2::rANS Boost::program_options)
target_compile_definitions(${targetName} PRIVATE -DSOURCE_T=uint8_t)

o2_add_executable(rans-encode-decode-16
                  TARGETVARNAME targetName
                  SOURCES run/bin-encode-decode.cxx
                  PUBLIC_LINK_LIBRARIES O2::rANS Boost::program_options)
target_compile_definitions(${targetName} PRIVATE -DSOURCE_T=uint16_t)

o2_add_executable(rans-encode-decode-32
                  TARGETVARNAME targetName
                  SOURCES run/bin-encode-decode.cxx
                  PUBLIC_LINK_LIBRARIES O2::rANS Boost::program_options)
target_compile_definitions(${targetName} PRIVATE -DSOURCE_T=uint32_t)
