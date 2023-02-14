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

/// \file GeometryTGeo.cxx
/// \brief Implementation of the GeometryTGeo class
/// \author bogdan.vulpescu@clermont.in2p3.fr - adapted from ITS, 21.09.2017

#include "MFTTracking/TrackerConfig.h"

#include <fairlogger/Logger.h>

//__________________________________________________________________________
void o2::mft::TrackerConfig::initialize(const MFTTrackingParam& trkParam)
{
  /// initialize from MFTTrackingParam (command line configuration parameters)

  mMinTrackPointsLTF = trkParam.MinTrackPointsLTF;
  mMinTrackPointsCA = trkParam.MinTrackPointsCA;
  mMinTrackStationsLTF = trkParam.MinTrackStationsLTF;
  mMinTrackStationsCA = trkParam.MinTrackStationsCA;
  mLTFclsRCut = trkParam.LTFclsRCut;
  mLTFclsR2Cut = mLTFclsRCut * mLTFclsRCut;
  mROADclsRCut = trkParam.ROADclsRCut;
  mROADclsR2Cut = mROADclsRCut * mROADclsRCut;
  mRBins = trkParam.RBins;
  mPhiBins = trkParam.PhiBins;
  mRPhiBins = trkParam.RBins * trkParam.PhiBins;
  mZVtxMin = trkParam.ZVtxMin;
  mZVtxMax = trkParam.ZVtxMax;
  mRCutAtZmin = trkParam.rCutAtZmin;
  mLTFConeRadius = trkParam.LTFConeRadius;
  mCAConeRadius = trkParam.CAConeRadius;
  mFullClusterScan = trkParam.FullClusterScan;
  mTrueTrackMCThreshold = trkParam.TrueTrackMCThreshold;

  assert(mRPhiBins < constants::index_table::MaxRPhiBins && "Track finder binning overflow");
}

//__________________________________________________________________________
void o2::mft::TrackerConfig::initBinContainers()
{
  if (!mBins) {
    mBins = std::make_unique<BinContainer>();
  }
  if (!mBinsS) {
    mBinsS = std::make_unique<BinContainer>();
  }
}
