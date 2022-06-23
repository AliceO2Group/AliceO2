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

/// \file IndexTableUtils.cxx
/// \brief
///

#include "TMath.h"
#include "StrangenessTracking/IndexTableUtils.h"

namespace o2
{
namespace strangeness_tracking
{

int indexTableUtils::getEtaBin(float eta)
{
  float deltaEta = (maxEta - minEta) / (mEtaBins);
  int bEta = (eta - minEta) / deltaEta; // bins recentered to 0
  return bEta;
};

int indexTableUtils::getPhiBin(float phi)
{
  float deltaPhi = (maxPhi - minPhi) / (mPhiBins);
  int bPhi = (phi - minPhi) / deltaPhi; // bin recentered to 0
  return bPhi;
}

int indexTableUtils::getBinIndex(float eta, float phi)
{
  float deltaPhi = (maxPhi - minPhi) / (mPhiBins);
  float deltaEta = (maxEta - minEta) / (mEtaBins);
  int bEta = getEtaBin(eta);
  int bPhi = getPhiBin(phi);
  return bEta >= mEtaBins || bPhi >= mPhiBins ? mEtaBins * mPhiBins : bEta + mEtaBins * bPhi;
}

std::vector<int> indexTableUtils::getBinRect(float eta, float phi, float deltaEta, float deltaPhi)
{
  std::vector<int> idxVec;
  int centralBin = getBinIndex(eta, phi);
  if (centralBin == mPhiBins * mEtaBins) { // condition for overflows
    idxVec.push_back(centralBin);
    return idxVec;
  }
  int minEtaBin = TMath::Max(0, getEtaBin(eta - deltaEta));
  int maxEtaBin = getEtaBin(eta + deltaEta);
  int minPhiBin = TMath::Max(0, getPhiBin(phi - deltaPhi));
  int maxPhiBin = getPhiBin(phi + deltaPhi);

  for (int iPhi{minPhiBin}; iPhi <= maxPhiBin; iPhi++) {
    if (iPhi >= mPhiBins)
      break;
    for (int iEta{minEtaBin}; iEta <= maxEtaBin; iEta++) {
      if (iEta >= mEtaBins)
        break;
      idxVec.push_back(iEta + mEtaBins * iPhi);
    }
  }
  return idxVec;
};
} // namespace strangeness_tracking
} // namespace o2
