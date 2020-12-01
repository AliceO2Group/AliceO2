// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
o2::mft::TrackerConfig::TrackerConfig()
  : mMinTrackPointsLTF{5},
    mMinTrackPointsCA{4},
    mMinTrackStationsLTF{4},
    mMinTrackStationsCA{4},
    mLTFclsRCut{0.0100},
    mROADclsRCut{0.0400},
    mLTFseed2BinWin{3},
    mLTFinterBinWin{3},
    mRBins{50},
    mPhiBins{50},
    mRPhiBins{50 * 50},
    mRBinSize{(constants::index_table::RMax - constants::index_table::RMin) / 50.},
    mPhiBinSize{(constants::index_table::PhiMax - constants::index_table::PhiMin) / 50.},
    mInverseRBinSize{50. / (constants::index_table::RMax - constants::index_table::RMin)},
    mInversePhiBinSize{50. / (constants::index_table::PhiMax - constants::index_table::PhiMin)}
{
  /// default constructor
}

//__________________________________________________________________________
o2::mft::TrackerConfig::TrackerConfig(const TrackerConfig& conf)
  : mMinTrackPointsLTF(conf.mMinTrackPointsLTF),
    mMinTrackPointsCA(conf.mMinTrackPointsCA),
    mMinTrackStationsLTF(conf.mMinTrackStationsLTF),
    mMinTrackStationsCA(conf.mMinTrackStationsCA),
    mLTFclsRCut(conf.mLTFclsRCut),
    mROADclsRCut(conf.mROADclsRCut),
    mLTFseed2BinWin(conf.mLTFseed2BinWin),
    mLTFinterBinWin(conf.mLTFinterBinWin),
    mRBins(conf.mRBins),
    mPhiBins(conf.mPhiBins),
    mRPhiBins(conf.mRPhiBins),
    mRBinSize(conf.mRBinSize),
    mPhiBinSize(conf.mPhiBinSize),
    mInverseRBinSize(conf.mInverseRBinSize),
    mInversePhiBinSize(conf.mInversePhiBinSize)
{
  /// copy constructor
}

//__________________________________________________________________________
o2::mft::TrackerConfig& o2::mft::TrackerConfig::operator=(const TrackerConfig& conf)
{
  /// assign constructor

  if (this == &conf) {
    return *this;
  }

  mMinTrackPointsLTF = conf.mMinTrackPointsLTF;
  mMinTrackPointsCA = conf.mMinTrackPointsCA;
  mMinTrackStationsLTF = conf.mMinTrackStationsLTF;
  mMinTrackStationsCA = conf.mMinTrackStationsCA;
  mLTFclsRCut = conf.mLTFclsRCut;
  mROADclsRCut = conf.mROADclsRCut;
  mLTFseed2BinWin = conf.mLTFseed2BinWin;
  mLTFinterBinWin = conf.mLTFinterBinWin;
  mRBins = conf.mRBins;
  mPhiBins = conf.mPhiBins;
  mRPhiBins = conf.mRPhiBins;
  mRBinSize = conf.mRBinSize;
  mPhiBinSize = conf.mPhiBinSize;
  mInverseRBinSize = conf.mInverseRBinSize;
  mInversePhiBinSize = conf.mInversePhiBinSize;
}

//__________________________________________________________________________
void o2::mft::TrackerConfig::initialize(const MFTTrackingParam& trkParam)
{
  /// initialize from MFTTrackingParam (command line configuration parameters)

  mMinTrackPointsLTF = trkParam.MinTrackPointsLTF;
  mMinTrackPointsCA = trkParam.MinTrackPointsCA;
  mMinTrackStationsLTF = trkParam.MinTrackStationsLTF;
  mMinTrackStationsCA = trkParam.MinTrackStationsCA;
  mLTFclsRCut = trkParam.LTFclsRCut;
  mROADclsRCut = trkParam.ROADclsRCut;
  mLTFseed2BinWin = trkParam.LTFseed2BinWin;
  mLTFinterBinWin = trkParam.LTFinterBinWin;

  mRBins = trkParam.RBins;
  mPhiBins = trkParam.PhiBins;
  mRPhiBins = trkParam.RBins * trkParam.PhiBins;
  if (mRPhiBins > constants::index_table::MaxRPhiBins) {
    LOG(WARN) << "To many RPhiBins for this configuration!";
    mRPhiBins = constants::index_table::MaxRPhiBins;
    mRBins = sqrt(constants::index_table::MaxRPhiBins);
    mPhiBins = sqrt(constants::index_table::MaxRPhiBins);
    LOG(WARN) << "Using instead RBins " << mRBins << " and PhiBins " << mPhiBins;
  }
  mRBinSize = (constants::index_table::RMax - constants::index_table::RMin) / mRBins;
  mPhiBinSize = (constants::index_table::PhiMax - constants::index_table::PhiMin) / mPhiBins;
}
