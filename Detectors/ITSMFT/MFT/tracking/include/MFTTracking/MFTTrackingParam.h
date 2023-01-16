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
  Linear,
  Optimized // Parameter propagation with helix model; covariance propagation with quadratic model
};

// **
// ** Parameters for MFT tracking configuration
// **
struct MFTTrackingParam : public o2::conf::ConfigurableParamHelper<MFTTrackingParam> {
  Int_t trackmodel = MFTTrackModel::Optimized;
  double MFTRadLength = 0.042; // MFT average material budget within acceptance
  bool verbose = false;
  bool forceZeroField = false; // Force MFT tracking with B=0
  float alignResidual = 0.f;   // Increment cluster covariance to account for alignment residuals

  /// tracking algorithm (LTF and CA) parameters
  /// minimum number of points for a LTF track
  Int_t MinTrackPointsLTF = 5;
  /// minimum number of points for a CA track
  Int_t MinTrackPointsCA = 5;
  /// minimum number of detector stations for a LTF track
  Int_t MinTrackStationsLTF = 4;
  /// minimum number of detector stations for a CA track
  Int_t MinTrackStationsCA = 4;
  /// maximum distance for a cluster to be attached to a seed line (LTF)
  Float_t LTFclsRCut = 0.0100;
  /// maximum distance for a cluster to be attached to a seed line (CA road)
  Float_t ROADclsRCut = 0.0400;
  /// number of bins in r-direction
  Int_t RBins = 30;
  /// number of bins in phi-direction
  Int_t PhiBins = 120;
  /// Minimum z vertex position for conical search bin optimization
  Float_t ZVtxMin = -13.f; // cm
  /// Maximum z vertex position for conical search bin optimization
  Float_t ZVtxMax = 13.f; // cm
  /// Radial vertex position cut at ZVtxMin
  Float_t rCutAtZmin = 0.1; // cm
  /// Special version for TED shots and cosmics, with full scan of the clusters
  bool FullClusterScan = false;
  /// road for LTF algo : cylinder or cone (default)
  Bool_t LTFConeRadius = kFALSE;
  /// road for CA algo : cylinder or cone (default)
  Bool_t CAConeRadius = kFALSE;
  /// Minimum fraction of correct clusters MC labels to set True MC tracks
  Float_t TrueTrackMCThreshold = 0.8;

  // cuts to reject to low or too high mult events, or externally provided IRFrames
  float cutMultClusLow = 0;   /// reject ROF with estimated cluster mult. below this value (no cut if <0)
  float cutMultClusHigh = -1; /// reject ROF with estimated cluster mult. above this value (no cut if <0)
  bool irFramesOnly = false;  ///< track only ROFs that overlap one of the IRFrames (provided externally by ITS)

  bool isMultCutRequested() const { return cutMultClusLow >= 0.f && cutMultClusHigh > 0.f; };
  bool isPassingMultCut(float mult) const { return mult >= cutMultClusLow && (mult <= cutMultClusHigh || cutMultClusHigh <= 0.f); }

  O2ParamDef(MFTTrackingParam, "MFTTracking");
};

} // end namespace mft
} // end namespace o2

#endif // ALICEO2_MFT_TRACKINGPARAM_H_
