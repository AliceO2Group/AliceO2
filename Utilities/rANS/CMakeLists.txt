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

option(RANS_ENABLE_JSON "compile librans with rapidjson" ON)

o2_add_header_only_library(rANS
               TARGETVARNAME LIBRANS
               INTERFACE_LINK_LIBRARIES FairLogger::FairLogger Microsoft.GSL::GSL)
if(${RANS_ENABLE_JSON})
  target_compile_definitions(${LIBRANS} INTERFACE RANS_ENABLE_JSON)
  target_link_libraries(${LIBRANS} INTERFACE RapidJSON::RapidJSON)
endif()
               #target_compile_options(${LIBRANS} INTERFACE -mavx2 -mfma -march=native)
if(${ENABLE_VTUNE_PROFILER})
  target_compile_definitions(${LIBRANS} INTERFACE ENABLE_VTUNE_PROFILER)
  target_link_libraries(${LIBRANS} INTERFACE PkgConfig::Vtune)
endif()

o2_add_test(Utils
            NAME Utils
            SOURCES test/test_ransUtils.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_UTILS
            LABELS utils)
            target_compile_options(${TEST_UTILS} PRIVATE -march=native)

o2_add_test(Iterators
            NAME Iterators
            SOURCES test/test_ransIterators.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_ITERATORS
            LABELS utils)
            target_compile_options(${TEST_ITERATORS} PRIVATE -march=native)

o2_add_test(BitPtr
            NAME BitPtr
            SOURCES test/test_ransBitPtr.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_BITPTR
            LABELS utils)
            target_compile_options(${TEST_BITPTR} PRIVATE -march=native)

o2_add_test(Pack
            NAME Pack
            SOURCES test/test_ransPack.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_PACK
            LABELS utils)
            target_compile_options(${TEST_PACK} PRIVATE -march=native)

o2_add_test(SIMDEncoderKernels
            NAME SIMDEncoderKernels
            SOURCES test/test_ransSIMDEncoderKernels.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_SIMD_ENCODER_KERNELS
            LABELS utils)
            target_compile_options(${TEST_SIMD_ENCODER_KERNELS} PRIVATE -march=native)

o2_add_test(SIMD
            NAME SIMD
            SOURCES test/test_ransSIMD.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_SIMD
            LABELS utils)
            target_compile_options(${TEST_SIMD} PRIVATE -march=native)

o2_add_test(AlignedArray
            NAME AlignedArray
            SOURCES test/test_ransAlignedArray.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_ALIGNED_ARRAY
            LABELS utils)
            target_compile_options(${TEST_ALIGNED_ARRAY} PRIVATE -march=native)

o2_add_test(HistogramView
            NAME HistogramView
            SOURCES test/test_ransHistogramView.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_HISTOGRAM_VIEW
            LABELS utils)
            target_compile_options(${TEST_HISTOGRAM_VIEW} PRIVATE -march=native)

o2_add_test(Histograms
            NAME Histograms
            SOURCES test/test_ransHistograms.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_HISTOGRAMS
            LABELS utils)
            target_compile_options(${TEST_HISTOGRAMS} PRIVATE -march=native)

o2_add_test(Metrics
            NAME Metrics
            SOURCES test/test_ransMetrics.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_METRICS
            LABELS utils)
            target_compile_options(${TEST_METRICS} PRIVATE -march=native)

o2_add_test(SymbolTables
            NAME SymbolTables
            SOURCES test/test_ransSymbolTables.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_SYMBOL_TABLES
            LABELS utils)
            target_compile_options(${TEST_SYMBOL_TABLES} PRIVATE -march=native)

o2_add_test(ReverseSymbolLookupTable
            NAME ReverseSymbolLookupTable
            SOURCES test/test_ransReverseSymbolLookupTable.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_REVERSE_SYMBOL_LOOKUP_TABLE
            LABELS utils)
            target_compile_options(${TEST_REVERSE_SYMBOL_LOOKUP_TABLE} PRIVATE -march=native)

o2_add_test(EncodeDecode
            NAME EncodeDecode
            SOURCES test/test_ransEncodeDecode.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_ENCODE_DECODE
            LABELS utils)
            target_compile_options(${TEST_ENCODE_DECODE} PRIVATE -march=native)

o2_add_test(Serialize
            NAME Serialize
            SOURCES test/test_ransSerialize.cxx
            PUBLIC_LINK_LIBRARIES O2::rANS
            COMPONENT_NAME rANS
            TARGETVARNAME TEST_SERIALIZE
            LABELS utils)
            target_compile_options(${TEST_SERIALIZE} PRIVATE -march=native)
            target_link_libraries(${TEST_SERIALIZE} PRIVATE)

if(${RANS_ENABLE_JSON})
  if (TARGET benchmark::benchmark)

    o2_add_header_only_library(libransBenchmark
                  TARGETVARNAME LIB_RANS_BENCHMARK
                  INTERFACE_LINK_LIBRARIES O2::rANS RapidJSON::RapidJSON benchmark::benchmark)
    target_compile_options(${LIB_RANS_BENCHMARK} INTERFACE -O3 -march=native)
    target_link_libraries(${LIB_RANS_BENCHMARK} INTERFACE -ltbb)
    if(${ENABLE_VTUNE_PROFILER})
      target_compile_definitions(${LIB_RANS_BENCHMARK} INTERFACE ENABLE_VTUNE_PROFILER)
      target_link_libraries(${LIB_RANS_BENCHMARK} INTERFACE PkgConfig::Vtune)
    endif()

    o2_add_executable(CombinedIterator
                      SOURCES benchmarks/bench_ransCombinedIterator.cxx
                      COMPONENT_NAME rANS
                      IS_BENCHMARK
                      PUBLIC_LINK_LIBRARIES O2::libransBenchmark)

    o2_add_executable(Encode
                      SOURCES benchmarks/bench_ransEncode.cxx
                      COMPONENT_NAME rANS
                      IS_BENCHMARK
                      PUBLIC_LINK_LIBRARIES O2::libransBenchmark)

    o2_add_executable(EncodeImpl
                      SOURCES benchmarks/bench_ransEncodeImpl.cxx
                      COMPONENT_NAME rANS
                      IS_BENCHMARK
                      PUBLIC_LINK_LIBRARIES O2::libransBenchmark)

    o2_add_executable(Streaming
                    SOURCES benchmarks/bench_ransStreaming.cxx
                    COMPONENT_NAME rANS
                    IS_BENCHMARK
                    PUBLIC_LINK_LIBRARIES O2::libransBenchmark)

    o2_add_executable(Pack
                    SOURCES benchmarks/bench_ransPack.cxx
                    COMPONENT_NAME rANS
                    IS_BENCHMARK
                    PUBLIC_LINK_LIBRARIES O2::libransBenchmark)

    o2_add_executable(UnPack
                    SOURCES benchmarks/bench_ransUnPack.cxx
                    COMPONENT_NAME rANS
                    IS_BENCHMARK
                    PUBLIC_LINK_LIBRARIES O2::libransBenchmark)

    o2_add_executable(TPCEncodeDecode
                      SOURCES benchmarks/bench_ransTPC.cxx
                      COMPONENT_NAME rANS
                      IS_BENCHMARK
                      PUBLIC_LINK_LIBRARIES O2::libransBenchmark Boost::program_options)
  endif()
endif()

o2_add_executable(rans-encode-decode-8
                  TARGETVARNAME RANS_ENCODE_DECODE_8
                  SOURCES run/bin-encode-decode.cxx
                  PUBLIC_LINK_LIBRARIES O2::rANS Boost::program_options)
target_compile_definitions(${RANS_ENCODE_DECODE_8} PRIVATE -DSOURCE_T=uint8_t)
target_compile_options(${RANS_ENCODE_DECODE_8} PRIVATE -O3 -march=native -fopenmp)

o2_add_executable(rans-encode-decode-16
                  TARGETVARNAME RANS_ENCODE_DECODE_16
                  SOURCES run/bin-encode-decode.cxx
                  PUBLIC_LINK_LIBRARIES O2::rANS Boost::program_options)
target_compile_definitions(${RANS_ENCODE_DECODE_16} PRIVATE -DSOURCE_T=uint16_t)
target_compile_options(${RANS_ENCODE_DECODE_16} PRIVATE -O3 -march=native -fopenmp)

o2_add_executable(rans-encode-decode-32
                  TARGETVARNAME RANS_ENCODE_DECODE_32
                  SOURCES run/bin-encode-decode.cxx
                  PUBLIC_LINK_LIBRARIES O2::rANS Boost::program_options)
target_compile_definitions(${RANS_ENCODE_DECODE_32} PRIVATE -DSOURCE_T=uint32_t)
target_compile_options(${RANS_ENCODE_DECODE_32} PRIVATE -O3 -march=native -fopenmp)
