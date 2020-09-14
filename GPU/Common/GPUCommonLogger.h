// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonLogger.h
/// \author David Rohr

#ifndef GPUCOMMONFAIRLOGGER_H
#define GPUCOMMONFAIRLOGGER_H

#if defined(__OPENCL__)
#define LOG(...)
#define LOGF(...)

#elif defined(GPUCA_GPUCODE_DEVICE)
#define LOG(...)
#define LOGF(type, string, ...)         \
  {                                     \
    printf(string "\n", ##__VA_ARGS__); \
  }

#elif defined(GPUCA_STANDALONE) ||                    \
  defined(GPUCA_ALIROOT_LIB) ||                       \
  (!defined(__cplusplus) || __cplusplus < 201703L) || \
  (defined(__HIPCC__) && (!defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI == 0))
#include <iostream>
#include <cstdio>
#define LOG(type) std::cout
#define LOGF(type, string, ...)         \
  {                                     \
    printf(string "\n", ##__VA_ARGS__); \
  }

#else
#include <Framework/Logger.h>

#endif

#endif
