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

namespace o2::gpu
{
typedef uint32_t calink;
typedef uint32_t cahit;
struct cahit2 { cahit x, y; };
} // namespace o2::GPU

#endif //GPUDTPCEF_H
// clang-format on
