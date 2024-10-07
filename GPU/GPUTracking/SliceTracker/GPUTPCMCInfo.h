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

/// \file GPUTPCMCInfo.h
/// \author David Rohr

#ifndef GPUTPCMCINFO_H
#define GPUTPCMCINFO_H

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTPCMCInfo {
  int32_t charge;
  int8_t prim;
  int8_t primDaughters;
  int32_t pid;
  float x;
  float y;
  float z;
  float pX;
  float pY;
  float pZ;
  float genRadius;
#ifdef GPUCA_TPC_GEOMETRY_O2
  float t0;
#endif
};
struct GPUTPCMCInfoCol {
  uint32_t first;
  uint32_t num;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
