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

#ifndef ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAMEGPU_H
#define ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAMEGPU_H

#include "ITStrackingGPU/TimeFrameGPU.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsTRK/Cluster.h"
#include "DataFormatsTRK/ROFRecord.h"

#include <gsl/span>
#include <nlohmann/json.hpp>

class TTree;

namespace o2
{
namespace trk
{
class GeometryTGeo;

template <int nLayers = 11>
class TimeFrameGPU : public o2::its::gpu::TimeFrameGPU<nLayers>
{
 public:
  TimeFrameGPU() = default;
  ~TimeFrameGPU() override = default;

  int loadROFsFromHitTree(TTree* hitsTree, GeometryTGeo* gman, const nlohmann::json& config);

  int loadROFrameData(gsl::span<const o2::trk::ROFRecord> rofs,
                      gsl::span<const o2::trk::Cluster> clusters,
                      gsl::span<const unsigned char> patterns,
                      const dataformats::MCTruthContainer<MCCompLabel>* mcLabels = nullptr,
                      float yPlaneMLOT = 0.f);

  void getPrimaryVerticesFromMC(TTree* mcHeaderTree, int nRofs, Long64_t nEvents, int inROFpileup);
  void addTruthSeedingVertices(gsl::span<const o2::trk::ROFRecord> rofs);
};

} // namespace trk
} // namespace o2

#endif
