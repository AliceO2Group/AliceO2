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
///
/// \file VertexerTraitsGPU.h
/// \brief
/// \author matteo.concas@cern.ch

// #define VTX_DEBUG
#ifndef ITSTRACKINGGPU_VERTEXERTRAITSGPU_H_
#define ITSTRACKINGGPU_VERTEXERTRAITSGPU_H_

#include <vector>

#include "ITStracking/VertexerTraits.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Tracklet.h"

#include "ITStrackingGPU/TimeFrameGPU.h"

namespace o2::its
{

template <int nLayers>
class VertexerTraitsGPU final : public VertexerTraits<nLayers>
{
 public:
  void initialise(const TrackingParameters&, const int iteration = 0) final;
  void adoptTimeFrame(TimeFrame<nLayers>* tf) noexcept final;
  void computeTracklets(const int iteration = 0) final;
  void computeTrackletMatching(const int iteration = 0) final;
  void computeVertices(const int iteration = 0) final;
  void updateVertexingParameters(const std::vector<VertexingParameters>&, const TimeFrameGPUParameters&) final;

  bool isGPU() const noexcept final { return true; }
  const char* getName() const noexcept final { return "GPU"; }

 protected:
  gpu::TimeFrameGPU<nLayers>* mTimeFrameGPU;
  TimeFrameGPUParameters mTfGPUParams;
};

} // namespace o2::its

#endif
