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

/// \file GeometryTGeo.h
/// \brief Definition of the GeometryTGeo class
/// \author bogdan.vulpescu@clermont.in2p3.fr - adapted from ITS, 21.09.2017

#ifndef ALICEO2_MFT_TRACKERCONFIG_H_
#define ALICEO2_MFT_TRACKERCONFIG_H_

#include <mutex>
#include "MFTTracking/Constants.h"
#include "MFTTracking/MFTTrackingParam.h"

namespace o2
{
namespace mft
{

using namespace constants::mft;
using BinContainer = std::array<std::array<std::array<std::vector<Int_t>, constants::index_table::MaxRPhiBins>, (constants::mft::LayersNumber - 1)>, (constants::mft::LayersNumber - 1)>;
using RArray = std::array<Float_t, constants::mft::LayersNumber>;
using PhiArray = std::array<Int_t, constants::mft::LayersNumber>;

class TrackerConfig
{
 public:
  TrackerConfig() = default;
  TrackerConfig(const TrackerConfig& conf) = default;
  TrackerConfig& operator=(const TrackerConfig& conf) = default;

  void initialize(const MFTTrackingParam& trkParam);

  const Int_t getRBinIndex(const Float_t r, const Int_t layer) const;
  const Int_t getPhiBinIndex(const Float_t phi) const;
  const Int_t getBinIndex(const Int_t rIndex, const Int_t phiIndex) const;

  static void initBinContainers();

  const std::pair<Int_t, Int_t>& getClusterBinIndexRange(Int_t layerId, Int_t bin) const { return mClusterBinIndexRange[layerId][bin]; }

 protected:
  // tracking configuration parameters
  Int_t mMinTrackPointsLTF{};
  Int_t mMinTrackPointsCA{};
  Int_t mMinTrackStationsLTF{};
  Int_t mMinTrackStationsCA{};
  Float_t mLTFclsRCut{};
  Float_t mLTFclsR2Cut{};
  Float_t mROADclsRCut{};
  Float_t mROADclsR2Cut{};
  Int_t mRBins{};
  Int_t mPhiBins{};
  Int_t mRPhiBins{};
  Float_t mZVtxMin{};
  Float_t mZVtxMax{};
  Float_t mRCutAtZmin{};
  Bool_t mLTFConeRadius{};
  Bool_t mCAConeRadius{};
  /// Special track finder for TED shots and cosmics, with full scan of the clusters
  bool mFullClusterScan = false;
  Float_t mTrueTrackMCThreshold{}; // Minimum fraction of correct MC labels to tag True tracks

  static std::mutex sTCMutex;

  static Float_t mPhiBinSize;
  static Float_t mInversePhiBinSize;

  static RArray mInverseRBinSize;
  static RArray mRBinSize;
  static PhiArray mPhiBinWin;

  static std::unique_ptr<BinContainer> mBins;
  static std::unique_ptr<BinContainer> mBinsS;
  std::array<std::array<std::pair<Int_t, Int_t>, constants::index_table::MaxRPhiBins>, constants::mft::LayersNumber> mClusterBinIndexRange{};

  ClassDefNV(TrackerConfig, 3);
};

inline Float_t TrackerConfig::mPhiBinSize;
inline Float_t TrackerConfig::mInversePhiBinSize;

inline RArray TrackerConfig::mInverseRBinSize;
inline RArray TrackerConfig::mRBinSize;
inline PhiArray TrackerConfig::mPhiBinWin;
inline std::unique_ptr<BinContainer> TrackerConfig::mBins;
inline std::unique_ptr<BinContainer> TrackerConfig::mBinsS;

inline const Int_t TrackerConfig::getRBinIndex(const Float_t r, const Int_t layer) const
{
  return (Int_t)((r - constants::index_table::RMin[layer]) * mInverseRBinSize[layer]);
}

inline const Int_t TrackerConfig::getPhiBinIndex(const Float_t phi) const
{
  return (Int_t)((phi - constants::index_table::PhiMin) * mInversePhiBinSize);
}

inline const Int_t TrackerConfig::getBinIndex(const Int_t rIndex, const Int_t phiIndex) const
{
  if (0 <= rIndex && rIndex < mRBins && 0 <= phiIndex && phiIndex < mPhiBins) {
    return (phiIndex * mRBins + rIndex);
  }
  return (mRBins * mPhiBins);
}

} // namespace mft
} // namespace o2

#endif
