// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDefMacros.h
/// \author David Rohr

// clang-format off
#ifndef GPUDEFMACROS_H
#define GPUDEFMACROS_H

#define GPUCA_M_EXPAND(x) x
#define GPUCA_M_STRIP_A(...) __VA_ARGS__
#define GPUCA_M_STRIP(X) GPUCA_M_STRIP_A X

#define GPUCA_M_STR_X(a) #a
#define GPUCA_M_STR(a) GPUCA_M_STR_X(a)

#define GPUCA_M_CAT_A(a, b) a ## b
#define GPUCA_M_CAT(...) GPUCA_M_CAT_A(__VA_ARGS__)
#define GPUCA_M_CAT3_A(a, b, c) a ## b ## c
#define GPUCA_M_CAT3(...) GPUCA_M_CAT3_A(__VA_ARGS__)

#define GPUCA_M_FIRST_A(a, ...) a
#define GPUCA_M_FIRST(...) GPUCA_M_FIRST_A(__VA_ARGS__)
#define GPUCA_M_SHIFT_A(a, ...) __VA_ARGS__
#define GPUCA_M_SHIFT(...) GPUCA_M_SHIFT_A(__VA_ARGS__)
#define GPUCA_M_FIRST2_A(a, b, ...) a, b
#define GPUCA_M_FIRST2(...) GPUCA_M_FIRST2_A(__VA_ARGS__, 0)

#define GPUCA_M_STRIP_FIRST(a) GPUCA_M_FIRST(GPUCA_M_STRIP(a))

#define GPUCA_M_COUNT_A(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) N
#define GPUCA_M_COUNT3_A(_1, _2, _3, N, ...) N
#define GPUCA_M_COUNT(...) GPUCA_M_COUNT_A(__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define GPUCA_M_SINGLEOPT(...) GPUCA_M_COUNT_A(__VA_ARGS__, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1)
#define GPUCA_M_MAX2_3(...) GPUCA_M_COUNT3_A(__VA_ARGS__, GPUCA_M_FIRST2(__VA_ARGS__), GPUCA_M_FIRST2(__VA_ARGS__), GPUCA_M_FIRST(__VA_ARGS__), )
#define GPUCA_M_MAX1_3(...) GPUCA_M_COUNT3_A(__VA_ARGS__, GPUCA_M_FIRST(__VA_ARGS__), GPUCA_M_FIRST(__VA_ARGS__), GPUCA_M_FIRST(__VA_ARGS__), )

#define GPUCA_M_UNROLL_
#define GPUCA_M_UNROLL_U(...) _Pragma(GPUCA_M_STR(unroll __VA_ARGS__))
#ifndef GPUCA_UNROLL
#define GPUCA_UNROLL(...)
#endif

#if !defined(WITH_OPENMP) || defined(GPUCA_GPUCODE_DEVICE)
#define GPUCA_OPENMP(...)
#else
#define GPUCA_OPENMP(...) _Pragma(GPUCA_M_STR(omp __VA_ARGS__))
#endif

#endif
// clang-format on
