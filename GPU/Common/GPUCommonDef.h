// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonDef.h
/// \author David Rohr

#ifndef GPUCOMMONDEF_H
#define GPUCOMMONDEF_H

// clang-format off
#if defined(__CINT__) || defined(__ROOTCINT__)
  #define CON_DELETE
  #define CON_DEFAULT
  #define CONSTEXPR const
#else
  #define CON_DELETE = delete
  #define CON_DEFAULT = default
  #define CONSTEXPR constexpr
#endif

#include "GPUCommonDefGPU.h"

#if (!defined(__OPENCL__) || defined(__OPENCLCPP__)) && !defined(GPUCA_ALIROOT_LIB)
  #define GPUCA_BUILD_MERGER
  #if defined(HAVE_O2HEADERS) && !defined(__HIPCC__)
    #define GPUCA_BUILD_TRD
  #endif
  #if defined(HAVE_O2HEADERS) && !defined(__HIPCC__)
    #define GPUCA_BUILD_ITS
  #endif
#endif

#if defined(GPUCA_STANDALONE) || defined(GPUCA_O2_LIB) || defined(GPUCA_GPULIBRARY)
  #define GPUCA_ALIGPUCODE
#endif
#ifdef GPUCA_ALIROOT_LIB
  #define GPUCA_NAMESPACE AliGPU
#else
  #define GPUCA_NAMESPACE o2
#endif
// clang-format on

#endif
