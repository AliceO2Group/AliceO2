// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterData.h
/// \author Matthias Kretz, Sergey Gorbunov, David Rohr

#ifndef GPUTPCCLUSTERDATA_H
#define GPUTPCCLUSTERDATA_H

#include "GPUTPCDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTPCClusterData {
  int id;
  short row;
  short flags;
  float x;
  float y;
  float z;
  float amp;
#ifdef GPUCA_FULL_CLUSTERDATA
  float pad;
  float time;
  float ampMax;
  float sigmaPad2;
  float sigmaTime2;
#endif
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // CLUSTERDATA_H
