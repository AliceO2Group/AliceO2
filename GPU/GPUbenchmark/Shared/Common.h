// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Common.h
/// \author: mconcas@cern.ch

#ifndef GPU_BENCHMARK_COMMON_H
#define GPU_BENCHMARK_COMMON_H
// #if defined(__HIPCC__)
// #include "hip/hip_runtime.h"
// #endif

#include <iostream>
#include <iomanip>

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define failed(...)                       \
  printf("%serror: ", KRED);              \
  printf(__VA_ARGS__);                    \
  printf("\n");                           \
  printf("error: TEST FAILED\n%s", KNRM); \
  exit(EXIT_FAILURE);

#endif