// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GeometryTGeo.h
/// \brief Definition of the GeometryTGeo class
/// \author bogdan.vulpescu@clermont.in2p3.fr - adapted from ITS, 21.09.2017

#ifndef ALICEO2_MFT_TRACKERCONFIG_H_
#define ALICEO2_MFT_TRACKERCONFIG_H_

#include "MFTTracking/Constants.h"
#include "MFTTracking/MFTTrackingParam.h"

namespace o2
{
namespace mft
{
class TrackerConfig
{
 public:
  TrackerConfig();
  TrackerConfig(const TrackerConfig& conf) = default;
  TrackerConfig& operator=(const TrackerConfig& conf) = default;

  void initialize(const MFTTrackingParam& trkParam);

  const Int_t getRBinIndex(const Float_t r) const;
  const Int_t getPhiBinIndex(const Float_t phi) const;
  const Int_t getBinIndex(const Int_t rIndex, const Int_t phiIndex) const;

  // tracking configuration parameters
  Int_t mMinTrackPointsLTF = 5;
  Int_t mMinTrackPointsCA = 4;
  Int_t mMinTrackStationsLTF = 4;
  Int_t mMinTrackStationsCA = 4;
  Float_t mLTFclsRCut = 0.0100;
  Float_t mROADclsRCut = 0.0400;
  Int_t mLTFseed2BinWin = 3;
  Int_t mLTFinterBinWin = 3;
  Int_t mRBins = 50;
  Int_t mPhiBins = 50;
  Int_t mRPhiBins = 50 * 50;
  Float_t mRBinSize = (constants::index_table::RMax - constants::index_table::RMin) / 50.;
  Float_t mPhiBinSize = constants::index_table::PhiMax / 50.;
  Float_t mInverseRBinSize = 50. / (constants::index_table::RMax - constants::index_table::RMin);
  Float_t mInversePhiBinSize = 50. / constants::index_table::PhiMax;

 private:
  ClassDefNV(TrackerConfig, 1);
};

inline const Int_t TrackerConfig::getRBinIndex(const Float_t r) const
{
  return (Int_t)((r - constants::index_table::RMin) * mInverseRBinSize);
}

inline const Int_t TrackerConfig::getPhiBinIndex(const Float_t phi) const
{
  return (Int_t)((phi - constants::index_table::PhiMin) * mInversePhiBinSize);
}

inline const Int_t TrackerConfig::getBinIndex(const Int_t rIndex, const Int_t phiIndex) const
{
  if (0 <= rIndex && rIndex < mRBins &&
      0 <= phiIndex && phiIndex < mPhiBins) {
    return (phiIndex * mRBins + rIndex);
  }
  return (mRBins * mPhiBins);
}

} // namespace mft
} // namespace o2

#endif
