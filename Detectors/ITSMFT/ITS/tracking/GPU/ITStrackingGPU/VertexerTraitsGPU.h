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
#include <array>

#include "ITStracking/VertexerTraits.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Tracklet.h"

#include "ITStrackingGPU/TimeFrameGPU.h"

namespace o2
{
namespace its
{
class ROframe;

using constants::its2::InversePhiBinSize;

class VertexerTraitsGPU : public VertexerTraits
{
 public:
  VertexerTraitsGPU();
  ~VertexerTraitsGPU() override;
  void initialise(const TrackingParameters& trackingParams) override;
  void adoptTimeFrame(TimeFrame* tf) override;
  void computeTracklets() override;
  void computeTrackletMatching() override;
  void computeVertices() override;
  void computeVerticesHist();
  // void computeMCFiltering() override;

 protected:
  IndexTableUtils* mDeviceIndexTableUtils;
  gpu::TimeFrameGPU<7>* mTimeFrameGPU;
};

inline void VertexerTraitsGPU::adoptTimeFrame(TimeFrame* tf) { mTimeFrameGPU = static_cast<gpu::TimeFrameGPU<7>*>(tf); }

} // namespace its
} // namespace o2
#endif
