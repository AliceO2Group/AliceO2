// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  int GenerateEvent(const GPUParam& sliceParam, char* filename) { return 1; }
  void FinishEventGenerator() {}

  static void RunEventGenerator(GPUChainTracking* rec){};
};

#else

class genEvents
{
 public:
  genEvents(GPUChainTracking* rec) : mRec(rec) {}
  void InitEventGenerator();
  int GenerateEvent(const GPUParam& sliceParam, char* filename);
  void FinishEventGenerator();

  static void RunEventGenerator(GPUChainTracking* rec);

 private:
  int GetSlice(double GlobalPhi);
  int GetDSlice(double LocalPhi);
  double GetSliceAngle(int iSlice);
  int RecalculateSlice(GPUTPCGMPhysicalTrackModel& t, int& iSlice);
  double GetGaus(double sigma);

  TH1F* mClusterError[3][2] = {{0, 0}, {0, 0}, {0, 0}};

  struct GenCluster {
    int sector;
    int row;
    int mcID;
    float x;
    float y;
    float z;
    unsigned int id;
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
