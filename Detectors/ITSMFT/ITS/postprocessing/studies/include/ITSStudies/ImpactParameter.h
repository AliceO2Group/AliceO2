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

#ifndef O2_IMPACT_PARAMETER_STUDY_H
#define O2_IMPACT_PARAMETER_STUDY_H

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace its
{
namespace study
{
using mask_t = o2::dataformats::GlobalTrackID::mask_t;

o2::framework::DataProcessorSpec getImpactParameterStudy(mask_t srcTracksMask, mask_t srcClusMask, bool useMC = false);
} // namespace study
} // namespace its
} // namespace o2

#endif // O2_IMPACT_PARAMETER_STUDY_H