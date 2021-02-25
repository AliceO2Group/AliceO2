// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_VERTEX_TRACK_MATCHER_SPEC_H
#define O2_VERTEX_TRACK_MATCHER_SPEC_H

/// @file VertexTrackMatcherSpec.h
/// @brief Specs for vertex track association device
/// @author ruben.shahoyan@cern.ch

#include "DetectorsVertexing/VertexTrackMatcher.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TStopwatch.h"

namespace o2
{
namespace vertexing
{

using namespace o2::framework;

class VertexTrackMatcherSpec : public Task
{
 public:
  VertexTrackMatcherSpec() = default;
  ~VertexTrackMatcherSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  o2::vertexing::VertexTrackMatcher mMatcher;
  TStopwatch mTimer;
};

/// create a processor spec
DataProcessorSpec getVertexTrackMatcherSpec(o2::detectors::DetID::mask_t dets);

} // namespace vertexing
} // namespace o2

#endif
