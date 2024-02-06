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

/// \file GPUTPCDef.h
/// \author David Rohr, Sergey Gorbunov

// clang-format off
#ifndef GPUTPCDEF_H
#define GPUTPCDEF_H

#include "GPUDef.h"

#define CALINK_INVAL ((calink) -1)
#define CALINK_DEAD_CHANNEL ((calink) -2)

namespace GPUCA_NAMESPACE
{
namespace gpu
{
#if defined(GPUCA_O2_LIB) || defined(GPUCA_O2_INTERFACE)
typedef unsigned int calink;
typedef unsigned int cahit;
#else
typedef unsigned int calink;
typedef unsigned int cahit;
#endif
struct cahit2 { cahit x, y; };
}
} // GPUCA_NAMESPACE::GPU

#ifdef GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME // Needs full clusterdata
  #define GPUCA_FULL_CLUSTERDATA
#endif

#if defined(GPUCA_STANDALONE) || defined(GPUCA_GPUCODE) // No support for Full Field Propagator or Statistical errors
  #ifdef GPUCA_GM_USE_FULL_FIELD
    #undef GPUCA_GM_USE_FULL_FIELD
  #endif
#endif

#endif //GPUDTPCEF_H
// clang format on
