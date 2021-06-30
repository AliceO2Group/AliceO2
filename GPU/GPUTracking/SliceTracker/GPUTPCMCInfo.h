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
  int charge;
  char prim;
  char primDaughters;
  int pid;
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
  unsigned int first;
  unsigned int num;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
