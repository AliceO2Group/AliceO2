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

#ifndef O2_TRACKINGITS_VERTEX_H_
#define O2_TRACKINGITS_VERTEX_H_

#include "GPUCommonDef.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <type_traits>
#endif
#include "ReconstructionDataFormats/Vertex.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsITS/TimeEstBC.h"

namespace o2::its
{
// NOTE: this uses the internal asymmetrical time reprenstation!
using Vertex = o2::dataformats::Vertex<o2::its::TimeEstBC>;
using VertexLabel = std::pair<o2::MCCompLabel, float>;
} // namespace o2::its

#ifndef GPUCA_GPUCODE_DEVICE
/// Defining ITS Vertex explicitly as messageable
namespace o2::framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::dataformats::Vertex<o2::its::TimeEstBC>> : std::true_type {
};
} // namespace o2::framework
#endif

#endif
