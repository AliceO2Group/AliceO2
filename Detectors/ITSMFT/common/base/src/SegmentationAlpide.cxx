// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SegmentationAlpide.cxx
/// \brief Implementation of the SegmentationAlpide class

#include "ITSMFTBase/SegmentationAlpide.h"
#include <cstdio>

ClassImp(o2::ITSMFT::SegmentationAlpide);

using namespace o2::ITSMFT;

void SegmentationAlpide::print()
{
  printf("Pixel size: %.2f (along %d rows) %.2f (along %d columns) microns\n", PitchRow*1e4,NRows,PitchCol*1e4,NCols);
  printf("Passive edges: bottom: %.2f, top: %.2f, left/right: %.2f microns\n",
         PassiveEdgeReadOut*1e4,PassiveEdgeTop*1e4,PassiveEdgeSide*1e4);
  printf("Active/Total size: %.6f/%.6f (rows) %.6f/%.6f (cols) cm\n",ActiveMatrixSizeRows,SensorSizeRows,
         ActiveMatrixSizeCols,SensorSizeCols);
}
