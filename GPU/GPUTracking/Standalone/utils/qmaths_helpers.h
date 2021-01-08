// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file qmaths_helpers.h
/// \author David Rohr

#ifndef QMATH_HELPERS_H
#define QMATH_HELPERS_H

#if defined __has_include
#if __has_include(<xmmintrin.h>) && __has_include(<pmmintrin.h>)
#include <xmmintrin.h>
#include <pmmintrin.h>
#if defined(_MM_FLUSH_ZERO_OFF) && defined(_MM_DENORMALS_ZERO_ON)
static void disable_denormals()
{
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}
#define XMM_HAS_DENORMAL_DEACTIVATE
#endif
#endif
#endif
#ifdef XMM_HAS_DENORMAL_DEACTIVATE
#undef XMM_HAS_DENORMAL_DEACTIVATE
#else
static void disable_denormals() {}
#endif

#endif
