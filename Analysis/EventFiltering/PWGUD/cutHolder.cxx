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

#include "cutHolder.h"

// setter
void cutHolder::SetisRun2(bool isRun2)
{
  misRun2 = isRun2;
}
void cutHolder::SetisMC(bool isMC)
{
  misMC = isMC;
}
void cutHolder::SetNTracks(int MinNTracks, int MaxNTracks)
{
  mMinNTracks = MinNTracks;
  mMaxNTracks = MaxNTracks;
}
void cutHolder::SetMinNTracksWithTOFHit(int MinNTracksWithTOFHit)
{
  mMinNTracksWithTOFHit = MinNTracksWithTOFHit;
}
void cutHolder::SetDeltaBC(int deltaBC)
{
  mdeltaBC = deltaBC;
}
void cutHolder::SetPoszRange(float MinPosz, float MaxPosz)
{
  mMinVertexPosz = MinPosz;
  mMaxVertexPosz = MaxPosz;
}

void cutHolder::SetPtRange(float minPt, float maxPt)
{
  mMinPt = minPt;
  mMaxPt = maxPt;
}
void cutHolder::SetEtaRange(float minEta, float maxEta)
{
  mMinEta = minEta;
  mMaxEta = maxEta;
}
void cutHolder::SetMaxTOFChi2(float maxTOFChi2)
{
  mMaxTOFChi2 = maxTOFChi2;
}
void cutHolder::SetMaxnSigmaTPC(float maxnSigma)
{
  mMaxnSigmaTPC = maxnSigma;
}
void cutHolder::SetMaxnSigmaTOF(float maxnSigma)
{
  mMaxnSigmaTOF = maxnSigma;
}

// getter
bool cutHolder::isRun2() const { return misRun2; }
bool cutHolder::isMC() const { return misMC; }
int cutHolder::minNTracks() const { return mMinNTracks; }
int cutHolder::maxNTracks() const { return mMaxNTracks; }
int cutHolder::minNTracksWithTOFHit() const { return mMinNTracksWithTOFHit; }
int cutHolder::deltaBC() const { return mdeltaBC; }
float cutHolder::minPosz() const { return mMinVertexPosz; }
float cutHolder::maxPosz() const { return mMaxVertexPosz; }
float cutHolder::minPt() const { return mMinPt; }
float cutHolder::maxPt() const { return mMaxPt; }
float cutHolder::minEta() const { return mMinEta; }
float cutHolder::maxEta() const { return mMaxEta; }
float cutHolder::maxTOFChi2() const { return mMaxTOFChi2; }
float cutHolder::maxnSigmaTPC() const { return mMaxnSigmaTPC; };
float cutHolder::maxnSigmaTOF() const { return mMaxnSigmaTOF; };
