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

/// \file GPUTPCGMOfflineFitter.h
/// \author Sergey Gorbunov

#ifndef GPUTPCGMOfflineFitter_H
#define GPUTPCGMOfflineFitter_H

#if (defined(GPUCA_ALIROOT_LIB) && !defined(GPUCA_GPUCODE))

#include "GPUParam.h"
#include "AliTPCtracker.h"

class GPUTPCGMMergedTrack;
class GPUTPCGMMergedTrackHit;
class AliTPCclusterMI;
class GPUTPCGMPolynomialField;

class GPUTPCGMOfflineFitter : public AliTPCtracker
{
 public:
  GPUTPCGMOfflineFitter();
  ~GPUTPCGMOfflineFitter();

  void Initialize(const GPUParam& hltParam, long TimeStamp, bool isMC);

  void RefitTrack(GPUTPCGMMergedTrack& track, const GPUTPCGMPolynomialField* field, GPUTPCGMMergedTrackHit* clusters);

  int32_t CreateTPCclusterMI(const GPUTPCGMMergedTrackHit& h, AliTPCclusterMI& c);

  bool FitOffline(const GPUTPCGMPolynomialField* field, GPUTPCGMMergedTrack& gmtrack, GPUTPCGMMergedTrackHit* clusters, int32_t& N);

 private:
  GPUParam fCAParam;
};

#endif

#endif
