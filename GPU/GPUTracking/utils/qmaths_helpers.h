// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file qmaths_helpers.h
/// \author David Rohr

#ifndef QMATH_HELPERS_H
#define QMATH_HELPERS_H

#if !(defined(__ARM_NEON) || defined(__aarch64__)) && __has_include(<xmmintrin.h>) // clang-format off
  #include <xmmintrin.h>
  #if __has_include(<pmmintrin.h>)
    #include <pmmintrin.h>
  #endif
#elif __has_include(<cfenv>)
  #include <cfenv>
#endif

static void disable_denormals()
{
#if !(defined(__ARM_NEON) || defined(__aarch64__)) && __has_include(<xmmintrin.h>)
  #if defined(_MM_FLUSH_ZERO_OFF) && defined(_MM_DENORMALS_ZERO_ON)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  #else
    #ifndef _MM_FLUSH_ZERO_ON
      #define _MM_FLUSH_ZERO_ON 0x8000
    #endif
    #ifndef _MM_DENORMALS_ZERO_ON
      #define _MM_DENORMALS_ZERO_ON 0x0040
    #endif
    _mm_setcsr(_mm_getcsr() | (_MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON));
  #endif
#elif __has_include(<cfenv>) && defined(FE_DFL_DISABLE_SSE_DENORMS_ENV)
  fesetenv(FE_DFL_DISABLE_SSE_DENORMS_ENV);
#endif // clang-format on
}

#endif
