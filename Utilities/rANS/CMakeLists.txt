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

o2_add_library(rANS
               TARGETVARNAME targetName
               SOURCES src/FrequencyTable.cxx
                       src/RenormedFrequencyTable.cxx
               PUBLIC_LINK_LIBRARIES FairLogger::FairLogger)
#target_compile_definitions(${targetName} PUBLIC O2_RANS_PRINT_PROCESSED_DATA)
target_compile_options(${targetName} PUBLIC -mavx2 -mfma -march=native)
if(${ENABLE_VTUNE_PROFILER})
  target_compile_definitions(${targetName} PUBLIC ENABLE_VTUNE_PROFILER)
  target_link_libraries(${targetName} PUBLIC PkgConfig::Vtune)
endif()

o2_add_test(SIMDEncoder
            NAME SIMDEncoder
            SOURCES test/test_SIMDKernel.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            LABELS utils)

o2_add_test(SIMDTypes
            NAME SIMDTypes
            SOURCES test/test_SIMDTypes.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            LABELS utils)

o2_add_test(Iterators
            NAME Iterators
            SOURCES test/test_ransIterators.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            LABELS utils)

o2_add_test(HistogramView
            NAME HistogramView
            SOURCES test/test_ransHistogramView.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            LABELS utils)

o2_add_test(FrequencyTable
            NAME FrequencyTable
            SOURCES test/test_ransFrequencyTable.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            LABELS utils)

o2_add_test(SymbolTable
            NAME SymbolTable
            SOURCES test/test_ransSymbolTable.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            LABELS utils)

o2_add_test(ReverseSymbolLookupTable
            NAME ReverseSymbolLookupTable
            SOURCES test/test_ransReverseSymbolLookupTable.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            LABELS utils)

o2_add_test(EncodeDecode
            NAME EncodeDecode
            SOURCES test/test_ransEncodeDecode.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            LABELS utils)

o2_add_test(EncodeDecodeLargeScale
            NAME EncodeDecodeLargeScale
            SOURCES test/test_ransEncodeDecodeLargeScale.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            LABELS utils)

if (TARGET benchmark::benchmark)
o2_add_executable(CombinedIterator
                  SOURCES benchmarks/bench_ransCombinedIterator.cxx
                  COMPONENT_NAME rANS
                  IS_BENCHMARK
                  PUBLIC_LINK_LIBRARIES O2::rANS benchmark::benchmark)

o2_add_executable(Compression
                  SOURCES benchmarks/bench_ransCompression.cxx
                  COMPONENT_NAME rANS
                  IS_BENCHMARK
                  TARGETVARNAME ransCompressionBenchmark
                  PUBLIC_LINK_LIBRARIES O2::rANS benchmark::benchmark)
  target_compile_options(${ransCompressionBenchmark} PRIVATE -O3  -march=native -fopenmp)
  target_link_libraries(${ransCompressionBenchmark} PRIVATE -ltbb)

o2_add_executable(Encoder
                  SOURCES benchmarks/bench_ransEncoder.cxx
                  COMPONENT_NAME rANS
                  IS_BENCHMARK
                  TARGETVARNAME ransEncoderBenchmark
                  PUBLIC_LINK_LIBRARIES O2::rANS benchmark::benchmark)
target_compile_options(${ransEncoderBenchmark} PRIVATE  -O3 -march=native -fopenmp)
target_link_libraries(${ransEncoderBenchmark} PRIVATE -ltbb)

o2_add_executable(LiteralEncoder
                  SOURCES benchmarks/bench_ransLiteralEncoder.cxx
                  COMPONENT_NAME rANS
                  IS_BENCHMARK
                  TARGETVARNAME ransLiteralEncoderBenchmark
                  PUBLIC_LINK_LIBRARIES O2::rANS benchmark::benchmark)
target_compile_options(${ransLiteralEncoderBenchmark} PRIVATE  -O3 -march=native -fopenmp)
target_link_libraries(${ransLiteralEncoderBenchmark} PRIVATE -ltbb)

o2_add_executable(Encode
                  SOURCES benchmarks/bench_ransEncode.cxx
                  COMPONENT_NAME rANS
                  IS_BENCHMARK
                  TARGETVARNAME ransEncodeBenchmark
                  PUBLIC_LINK_LIBRARIES O2::rANS benchmark::benchmark)
target_compile_options(${ransEncodeBenchmark} PRIVATE  -O3 -march=native -fopenmp)
target_link_libraries(${ransEncodeBenchmark} PRIVATE -ltbb)

o2_add_executable(SymbolTableAccess
                  SOURCES benchmarks/bench_ransSymbolTableAccess.cxx
                  COMPONENT_NAME rANS
                  IS_BENCHMARK
                  TARGETVARNAME ransSymbolTableAccessBenchmark
                  PUBLIC_LINK_LIBRARIES O2::rANS benchmark::benchmark)
target_compile_options(${ransSymbolTableAccessBenchmark} PRIVATE  -O3 -march=native -fopenmp)
target_link_libraries(${ransSymbolTableAccessBenchmark} PRIVATE -ltbb)

o2_add_executable(Streaming
                SOURCES benchmarks/bench_ransStreaming.cxx
                COMPONENT_NAME rANS
                IS_BENCHMARK
                TARGETVARNAME ransStreamingBenchmark
                PUBLIC_LINK_LIBRARIES O2::rANS benchmark::benchmark)
  target_compile_options(${ransStreamingBenchmark} PRIVATE  -O2 -march=native -fopenmp)
  target_link_libraries(${ransStreamingBenchmark} PRIVATE -ltbb)
  if(${ENABLE_VTUNE_PROFILER})
    target_compile_definitions(${ransStreamingBenchmark} PRIVATE ENABLE_VTUNE_PROFILER)
    target_link_libraries(${ransStreamingBenchmark} PRIVATE PkgConfig::Vtune)
  endif()


o2_add_executable(TPCEncodeDecode
                  SOURCES benchmarks/bench_ransTPC.cxx
                  COMPONENT_NAME rANS
                  IS_BENCHMARK
                  PUBLIC_LINK_LIBRARIES O2::rANS 
                  TARGETVARNAME ransTPCBenchmark
                  PUBLIC_LINK_LIBRARIES O2::rANS Boost::program_options FairLogger::FairLogger RapidJSON::RapidJSON)
  target_compile_options(${ransTPCBenchmark} PRIVATE  -O3 -march=native -fopenmp)
  target_link_libraries(${ransTPCBenchmark} PRIVATE -ltbb)
  if(${ENABLE_VTUNE_PROFILER})
  target_compile_definitions(${ransTPCBenchmark} PRIVATE ENABLE_VTUNE_PROFILER)
  target_link_libraries(${ransTPCBenchmark} PRIVATE PkgConfig::Vtune)
endif()

endif()

o2_add_executable(SIMD
                  SOURCES benchmarks/bench_ransSIMD.cxx
                  COMPONENT_NAME rANS
                  IS_BENCHMARK
                  TARGETVARNAME ransSIMDBenchmark
                  PUBLIC_LINK_LIBRARIES O2::rANS benchmark::benchmark)
target_compile_options(${ransSIMDBenchmark} PRIVATE  -O3 -march=native -fopenmp)
target_link_libraries(${ransSIMDBenchmark} PRIVATE -ltbb)

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