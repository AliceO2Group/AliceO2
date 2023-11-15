// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   defines.h
/// @author michael.lettrich@cern.ch
/// @brief  preprocessor defines to enable features based on CPU architecture

#ifndef RANS_INTERNAL_COMMON_DEFINES_H_
#define RANS_INTERNAL_COMMON_DEFINES_H_

#include <version>

#ifdef RANS_AVX2
#error RANS_AVX2 cannot be directly set
#endif
#ifdef RANS_SSE
#error RANS_SSE cannot be directly set
#endif
#ifdef RANS_SINGLE_STREAM
#error RANS_AVX cannot be directly set
#endif
#ifdef RANS_COMPAT
#error RANS_COMPAT cannot be directly set
#endif
#ifdef RANS_SIMD
#error RANS_SIMD cannot be directly set
#endif
#ifdef RANS_SSE_ONLY
#error RANS_SSE_ONLY cannot be directly set
#endif
#ifdef RANS_FMA
#error RANS_FMA cannot be directly set
#endif

#if (defined(__x86_64__) || defined(__aarch64__))
#define RANS_COMPAT
#if defined(__SIZEOF_INT128__)
#define RANS_SINGLE_STREAM
#endif // if defined(__SIZEOF_INT128__)
#endif // if 64 BIT system

#if defined(__x86_64__)
#if defined(__SSE4_2__)
#define RANS_SSE
#endif // SSE4.2
#if defined(__AVX2__)
#define RANS_AVX2
#endif // AVX2
#endif // x86

#if (defined(RANS_SSE) && !defined(RANS_AVX2))
#define RANS_SSE_ONLY
#endif

#if (defined(RANS_SSE) || defined(RANS_AVX2))
#define RANS_SIMD
#endif

#if defined(__FMA__)
#define RANS_FMA
#endif

#if defined(RANS_ENABLE_PARALLEL_STL) && defined(__cpp_lib_execution)
#define RANS_PARALLEL_STL
#endif

#endif /*RANS_INTERNAL_COMMON_DEFINES_H_*/