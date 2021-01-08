// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R. Pezzi - May 2020

#ifndef ALICEO2_MFT_TRACKINGPARAM_H_
#define ALICEO2_MFT_TRACKINGPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace mft
{

enum MFTTrackModel {
  Helix,
  Quadratic,
  Linear
};

// **
// ** Parameters for MFT tracking configuration
// **
struct MFTTrackingParam : public o2::conf::ConfigurableParamHelper<MFTTrackingParam> {
  Int_t trackmodel = MFTTrackModel::Helix;
  double MFTRadLength = 1.0; // MFT average material budget within acceptance. Should be 0.041
  bool verbose = false;

  /// tracking algorithm (LTF and CA) parameters
  /// minimum number of points for a LTF track
  Int_t MinTrackPointsLTF = 5;
  /// minimum number of points for a CA track
  Int_t MinTrackPointsCA = 4;
  /// minimum number of detector stations for a LTF track
  Int_t MinTrackStationsLTF = 4;
  /// minimum number of detector stations for a CA track
  Int_t MinTrackStationsCA = 4;
  /// maximum distance for a cluster to be attached to a seed line (LTF)
  Float_t LTFclsRCut = 0.0100;
  /// maximum distance for a cluster to be attached to a seed line (CA road)
  Float_t ROADclsRCut = 0.0400;
  /// number of bins in r-direction
  Int_t RBins = 50;
  /// number of bins in phi-direction
  Int_t PhiBins = 50;
  /// RPhi search window bin width for the second point of a seed (LTF and CA)
  Int_t LTFseed2BinWin = 3;
  /// RPhi search window bin width for the intermediate points
  Int_t LTFinterBinWin = 3;

  O2ParamDef(MFTTrackingParam, "MFTTracking");
};

} // end namespace mft
} // end namespace o2

#endif // ALICEO2_MFT_TRACKINGPARAM_H_
