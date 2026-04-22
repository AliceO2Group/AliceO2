// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ITS3_CLUSTERER_H
#define ALICEO2_ITS3_CLUSTERER_H

#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "ITS3Base/SegmentationMosaix.h"
#include "ITS3Reconstruction/LookUp.h"

namespace o2::its3
{
// Maximum number of rows is determined by Alpide
using Clusterer = o2::itsmft::ClustererT<o2::its3::LookUp, o2::itsmft::SegmentationAlpide::NRows>;
} // namespace o2::its3
extern template class o2::itsmft::ClustererT<o2::its3::LookUp, o2::itsmft::SegmentationAlpide::NRows>;

#endif
