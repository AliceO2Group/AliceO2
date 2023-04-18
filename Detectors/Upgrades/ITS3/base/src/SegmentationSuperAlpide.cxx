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

/// \file SegmentationSuperAlpide.cxx
/// \brief Implementation of the SegmentationSuperAlpide class

#include "ITS3Base/SegmentationSuperAlpide.h"
#include <Framework/Logger.h>

ClassImp(o2::its3::SegmentationSuperAlpide);

void o2::its3::SegmentationSuperAlpide::print()
{
  LOGP(debug, "SegmentationSuperAlpide:");
  LOGP(debug, "Layer {}: Active/Total size {:.2f}/{:.2f} (rows) x {:.2f}/{:.2f}", mLayer, mActiveMatrixSizeRows, mSensorSizeRows, mActiveMatrixSizeCols, mSensorSizeCols);
  LOGP(debug, "Pixel size: {:.2f} (along {} rows) {:.2f} (along {} columns) microns", mPitchRow * 1e4, mNRows, mPitchCol * 1e4, mNCols);
  LOGP(debug, "Passive edges: bottom: {:.6f}, top: {:.6f}, left/right: {:.6f} microns", mPassiveEdgeReadOut * 1e4, mPassiveEdgeTop * 1e4, mPassiveEdgeSide * 1e4);
}
