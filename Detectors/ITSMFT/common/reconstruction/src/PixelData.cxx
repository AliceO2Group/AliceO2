// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PixelData.cxx
/// \brief Implementation for transient data of single pixel and set of pixels from current chip

#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include <cassert>

using namespace o2::ITSMFT;

void PixelData::sanityCheck() const
{
  // make sure the mask used in this class are compatible with Alpide segmenations
  static_assert(RowMask + 1 >= o2::ITSMFT::SegmentationAlpide::NRows);
}
