// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPULogging.h
/// \author David Rohr

#ifndef GPULOGGING_H
#define GPULOGGING_H

#define GPUError(...)    \
  {                      \
    printf(__VA_ARGS__); \
    printf("\n");        \
  }
#define GPUWarning(...)  \
  {                      \
    printf(__VA_ARGS__); \
    printf("\n");        \
  }
#define GPUInfo(...)     \
  {                      \
    printf(__VA_ARGS__); \
    printf("\n");        \
  }
#define GPUImportant(...) \
  {                       \
    printf(__VA_ARGS__);  \
    printf("\n");         \
  }
#define GPUFatal(...)    \
  {                      \
    printf(__VA_ARGS__); \
    printf("\n");        \
    exit(1);             \
  }

#endif // GPULOGGING_H
