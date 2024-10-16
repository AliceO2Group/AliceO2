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

/// \file GPUTPCMCTrack.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCMCTRACK_H
#define GPUTPCMCTRACK_H

#include "GPUTPCDef.h"

class TParticle;

/**
 * @class GPUTPCMCTrack
 * store MC track information for GPUTPCPerformance
 */
class GPUTPCMCTrack
{
 public:
  GPUTPCMCTrack();
  GPUTPCMCTrack(const TParticle* part);

  void SetTPCPar(float X, float Y, float Z, float Px, float Py, float Pz);

  int32_t PDG() const { return fPDG; }
  const double* Par() const { return fPar; }
  const double* TPCPar() const { return fTPCPar; }
  double P() const { return fP; }
  double Pt() const { return fPt; }

  int32_t NHits() const { return mNHits; }
  int32_t NMCPoints() const { return fNMCPoints; }
  int32_t FirstMCPointID() const { return fFirstMCPointID; }
  int32_t NReconstructed() const { return fNReconstructed; }
  int32_t Set() const { return fSet; }
  int32_t NTurns() const { return fNTurns; }

  void SetP(float v) { fP = v; }
  void SetPt(float v) { fPt = v; }
  void SetPDG(int32_t v) { fPDG = v; }
  void SetPar(int32_t i, double v) { fPar[i] = v; }
  void SetTPCPar(int32_t i, double v) { fTPCPar[i] = v; }
  void SetNHits(int32_t v) { mNHits = v; }
  void SetNMCPoints(int32_t v) { fNMCPoints = v; }
  void SetFirstMCPointID(int32_t v) { fFirstMCPointID = v; }
  void SetNReconstructed(int32_t v) { fNReconstructed = v; }
  void SetSet(int32_t v) { fSet = v; }
  void SetNTurns(int32_t v) { fNTurns = v; }

 protected:
  int32_t fPDG;        //* particle pdg code
  double fPar[7];      //* x,y,z,ex,ey,ez,q/p
  double fTPCPar[7];   //* x,y,z,ex,ey,ez,q/p at TPC entrance (x=y=0 means no information)
  double fP, fPt;      //* momentum and transverse momentum
  int32_t mNHits;      //* N TPC clusters
  int32_t fNMCPoints;  //* N MC points
  int32_t fFirstMCPointID; //* id of the first MC point in the points array
  int32_t fNReconstructed; //* how many times is reconstructed
  int32_t fSet;            //* set of tracks 0-OutSet, 1-ExtraSet, 2-RefSet
  int32_t fNTurns;         //* N of turns in the current sector
};

#endif // GPUTPCMCTrack
