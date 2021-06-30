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

#ifndef O2_VERTEX_TRACK_MATCHER_SPEC_H
#define O2_VERTEX_TRACK_MATCHER_SPEC_H

/// @file VertexTrackMatcherSpec.h
/// @brief Specs for vertex track association device
/// @author ruben.shahoyan@cern.ch

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace vertexing
{

/// create a processor spec
o2::framework::DataProcessorSpec getVertexTrackMatcherSpec(o2::dataformats::GlobalTrackID::mask_t src);

} // namespace vertexing
} // namespace o2

#endif
