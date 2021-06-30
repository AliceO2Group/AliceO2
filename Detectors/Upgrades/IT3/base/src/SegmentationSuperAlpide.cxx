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
#include <cstdio>

ClassImp(o2::its3::SegmentationSuperAlpide);

using namespace o2::its3;

void SegmentationSuperAlpide::print()
{
  printf("ITS3 segmentation for layer: %d \n", mLayer);
  printf("Pixel size: %.2f (along %d rows) %.2f (along %d columns) microns\n", PitchRow * 1e4, NRows, PitchCol * 1e4, NCols);
  printf("Passive edges: bottom: %.2f, top: %.2f, left/right: %.2f microns\n",
         PassiveEdgeReadOut * 1e4, PassiveEdgeTop * 1e4, PassiveEdgeSide * 1e4);
  printf("Active/Total size: %.6f/%.6f (rows) %.6f/%.6f (cols) cm\n", ActiveMatrixSizeRows, SensorSizeRows,
         ActiveMatrixSizeCols, SensorSizeCols);
}
