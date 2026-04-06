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

#ifndef O2_ITS3_ALIGNMENT_H
#define O2_ITS3_ALIGNMENT_H

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/DataProcessorSpec.h"
#include "CommonUtils/EnumFlags.h"

namespace o2::its3::align
{

enum class OutputOpt : uint8_t {
  VerboseGBL = 0,
  MilleData,
  MilleSteer,
  MilleRes,
  Debug,
};
using OutputEnum = utils::EnumFlags<OutputOpt>;

o2::framework::DataProcessorSpec getAlignmentSpec(o2::dataformats::GlobalTrackID::mask_t srcTracks, o2::dataformats::GlobalTrackID::mask_t srcClus, bool useMC, bool withPV, bool withITS3, OutputEnum out);
} // namespace o2::its3::align

#endif
