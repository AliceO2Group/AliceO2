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

/// \file genEvents.h
/// \author Sergey Gorbunov

#ifndef GENEVENTS_H
#define GENEVENTS_H

#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUChainTracking;
struct GPUParam;
class GPUTPCGMPhysicalTrackModel;
#if !defined(GPUCA_BUILD_QA) || defined(_WIN32)
class genEvents
{
 public:
  genEvents(GPUChainTracking* rec) {}
  void InitEventGenerator() {}
  int32_t GenerateEvent(const GPUParam& sliceParam, char* filename) { return 1; }
  void FinishEventGenerator() {}

  static void RunEventGenerator(GPUChainTracking* rec){};
};

#else

class genEvents
{
 public:
  genEvents(GPUChainTracking* rec) : mRec(rec) {}
  void InitEventGenerator();
  int32_t GenerateEvent(const GPUParam& sliceParam, char* filename);
  void FinishEventGenerator();

  static void RunEventGenerator(GPUChainTracking* rec);

 private:
  int32_t GetSlice(double GlobalPhi);
  int32_t GetDSlice(double LocalPhi);
  double GetSliceAngle(int32_t iSlice);
  int32_t RecalculateSlice(GPUTPCGMPhysicalTrackModel& t, int32_t& iSlice);
  double GetGaus(double sigma);

  TH1F* mClusterError[3][2] = {{nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr}};

  struct GenCluster {
    int32_t sector;
    int32_t row;
    int32_t mcID;
    float x;
    float y;
    float z;
    uint32_t id;
  };

  const double mTwoPi = 2 * M_PI;
  const double mSliceDAngle = mTwoPi / 18.;
  const double mSliceAngleOffset = mSliceDAngle / 2;

  GPUChainTracking* mRec;
};

#endif
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
