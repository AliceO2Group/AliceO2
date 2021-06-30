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

#ifndef O2_PRIMARY_VERTEXER_SPEC_H
#define O2_PRIMARY_VERTEXER_SPEC_H

/// @file PrimaryVertexingSpec.h

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace vertexing
{

/// create a processor spec
o2::framework::DataProcessorSpec getPrimaryVertexingSpec(o2::dataformats::GlobalTrackID::mask_t src, bool validateWithFT0, bool useMC);

} // namespace vertexing
} // namespace o2

#endif
