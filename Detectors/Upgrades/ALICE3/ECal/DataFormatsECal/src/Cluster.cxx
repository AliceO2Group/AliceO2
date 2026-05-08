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

/// \file Cluster.cxx
/// \brief Implementation of ECal cluster class
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#include <map>
#include <vector>
#include <DataFormatsECal/Cluster.h>
#include <DataFormatsECal/Digit.h>

using namespace o2::ecal;

ClassImp(Cluster);

//==============================================================================
void Cluster::addDigit(int digitIndex, int towerId, double energy)
{
  mE += energy;
  mDigitIndex.push_back(digitIndex);
  mDigitTowerId.push_back(towerId);
  mDigitEnergy.push_back(energy);
}

//==============================================================================
int Cluster::getMcTrackID() const
{
  float maxEnergy = 0;
  int maxID = 0;
  for (const auto& [mcTrackID, energy] : mMcTrackEnergy) {
    if (energy > maxEnergy) {
      maxEnergy = energy;
      maxID = mcTrackID;
    }
  }
  return maxID;
}

//==============================================================================
TLorentzVector Cluster::getMomentum() const
{
  double r = std::sqrt(mX * mX + mY * mY + mZ * mZ);
  if (r == 0) {
    return TLorentzVector();
  }
  return TLorentzVector(mE * mX / r, mE * mY / r, mE * mZ / r, mE);
}
