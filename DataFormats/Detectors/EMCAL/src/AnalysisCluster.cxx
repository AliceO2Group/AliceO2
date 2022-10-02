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

/// \file AnalysisCluster.cxx

#include <fairlogger/Logger.h>
#include <gsl/span>
#include <array>
#include <TLorentzVector.h>
#include "DataFormatsEMCAL/AnalysisCluster.h"

using namespace o2::emcal;

//_______________________________________________________________________
void AnalysisCluster::clear()
{
  //if(mTracksMatched) delete mTracksMatched;
  //mTracksMatched = 0;
  mCellsAmpFraction.clear();
  mCellsIndices.clear();
}

//_______________________________________________________________________
TLorentzVector AnalysisCluster::getMomentum(std::array<const float, 3> vertex) const
{

  TLorentzVector p;

  float pos[3] = {mGlobalPos.X(), mGlobalPos.Y(), mGlobalPos.Z()};
  pos[0] -= vertex[0];
  pos[1] -= vertex[1];
  pos[2] -= vertex[2];

  float r = TMath::Sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);

  if (r > 0) {
    p.SetPxPyPzE(mEnergy * pos[0] / r, mEnergy * pos[1] / r, mEnergy * pos[2] / r, mEnergy);
  } else {
    LOG(info) << "Null cluster radius, momentum calculation not possible";
  }

  return p;
}

//______________________________________________________________________________
void AnalysisCluster::setGlobalPosition(math_utils::Point3D<float> x)
{
  mGlobalPos.SetX(x.X());
  mGlobalPos.SetY(x.Y());
  mGlobalPos.SetZ(x.Z());
}

//______________________________________________________________________________
void AnalysisCluster::setLocalPosition(math_utils::Point3D<float> x)
{
  mLocalPos.SetX(x.X());
  mLocalPos.SetY(x.Y());
  mLocalPos.SetZ(x.Z());
}
