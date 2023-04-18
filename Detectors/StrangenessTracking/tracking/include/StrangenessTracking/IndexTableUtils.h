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

/// \file IndexTableUtils.h
/// \brief
///
#ifndef STRTRACKING_INCLUDE_INDEXTABLEUTILS_H_
#define STRTRACKING_INCLUDE_INDEXTABLEUTILS_H_

#include "TMath.h"

namespace o2
{
namespace strangeness_tracking
{

struct IndexTableUtils {
  int getEtaBin(float eta);
  int getPhiBin(float phi);
  int getBinIndex(float eta, float phi);
  std::vector<int> getBinRect(float eta, float phi, float deltaEta, float deltaPhi);
  int mEtaBins = 64, mPhiBins = 64;
  float minEta = -1.5, maxEta = 1.5;
  float minPhi = 0., maxPhi = 2 * TMath::Pi();
};

inline int IndexTableUtils::getEtaBin(float eta)
{
  float deltaEta = (maxEta - minEta) / (mEtaBins);
  int bEta = (eta - minEta) / deltaEta; // bins recentered to 0
  return bEta;
};

inline int IndexTableUtils::getPhiBin(float phi)
{
  float deltaPhi = (maxPhi - minPhi) / (mPhiBins);
  int bPhi = (phi - minPhi) / deltaPhi; // bin recentered to 0
  return bPhi;
}

inline int IndexTableUtils::getBinIndex(float eta, float phi)
{
  float deltaPhi = (maxPhi - minPhi) / (mPhiBins);
  float deltaEta = (maxEta - minEta) / (mEtaBins);
  int bEta = getEtaBin(eta);
  int bPhi = getPhiBin(phi);
  return (bEta >= mEtaBins || bPhi >= mPhiBins || bEta < 0 || bPhi < 0) ? mEtaBins * mPhiBins : bEta + mEtaBins * bPhi;
}

inline std::vector<int> IndexTableUtils::getBinRect(float eta, float phi, float deltaEta, float deltaPhi)
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
    if (iPhi >= mPhiBins) {
      break;
    }
    for (int iEta{minEtaBin}; iEta <= maxEtaBin; iEta++) {
      if (iEta >= mEtaBins) {
        break;
      }
      idxVec.push_back(iEta + mEtaBins * iPhi);
    }
  }
  return idxVec;
};

} // namespace strangeness_tracking
} // namespace o2

#endif