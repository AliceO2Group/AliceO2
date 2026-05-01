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
/// \file TimeFrame.h
/// \brief TRK TimeFrame class derived from ITS TimeFrame
///

#ifndef ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAME_H
#define ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAME_H

#include "CommonDataFormat/InteractionRecord.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsTRK/Cluster.h"
#include "DataFormatsTRK/ROFRecord.h"
#include <array>
#include <gsl/span>
#include <vector>
#include <unordered_map>
#include <bitset>
#include <cstdint>

#include <nlohmann/json.hpp>

class TTree;

namespace o2
{
namespace trk
{
class GeometryTGeo;

/// TRK TimeFrame class that extends ITS TimeFrame functionality
/// This allows for customization of tracking algorithms specific to the TRK detector
template <int nLayers = 11>
class TimeFrame : public o2::its::TimeFrame<nLayers>
{
 public:
  TimeFrame() = default;
  ~TimeFrame() override = default;

  /// Override methods if needed for TRK-specific behavior
  /// For now, we inherit all functionality from ITS TimeFrame

  /// Process hits from TTree to initialize ROFs
  /// \param hitsTree Tree containing TRK hits
  /// \param mcHeaderTree Tree containing MC event headers
  /// \param nEvents Number of events to process
  /// \param gman TRK geometry manager instance
  /// \param config Configuration parameters for hit reconstruction
  int loadROFsFromHitTree(TTree* hitsTree, GeometryTGeo* gman, const nlohmann::json& config);

  /// Load ROF data from TRK clustered inputs (without topology dictionary for the time being).
  /// Patterns are expected as [rowSpan, colSpan, bitmap...] for each cluster.
  int loadROFrameData(gsl::span<const o2::trk::ROFRecord> rofs,
                      gsl::span<const o2::trk::Cluster> clusters,
                      gsl::span<const unsigned char> patterns,
                      const dataformats::MCTruthContainer<MCCompLabel>* mcLabels = nullptr,
                      float yPlaneMLOT = 0.f);

  /// Add primary vertices from MC headers for each ROF
  /// \param mcHeaderTree Tree containing MC event headers
  /// \param nRofs Number of ROFs (Read-Out Frames)
  /// \param nEvents Number of events to process
  /// \param inROFpileup Number of events per ROF
  void getPrimaryVerticesFromMC(TTree* mcHeaderTree, int nRofs, Long64_t nEvents, int inROFpileup);

  /// Add primary vertices using truth seeding from the DigitizationContext (collisioncontext.root).
  /// Maps each MC collision to its ROF via the ROF BCData timestamps (TRK digitising timing).
  /// \param rofs Span of TRK ROF records used to determine which ROF each collision falls into
  void addTruthSeedingVertices(gsl::span<const o2::trk::ROFRecord> rofs);

  /// Derive the per-layer LayerTiming from the per-layer ROF spans and initialise
  /// the ROF lookup tables. Each layer can have its own mROFLength and mROFBias,
  /// so staggered TRK readouts are handled naturally as long as the input
  /// ROFRecords carry the right BCData. The TF anchor (used to keep timing
  /// values bounded when expressed as BC offsets) is set to the earliest
  /// rofs[0].BCData across layers; consumers can read it back via getTFAnchorIR().
  /// Idempotent — must be called before loadROFrameData() in the cluster path.
  /// \param layerROFs One ROFRecord span per layer.
  void deriveAndInitTiming(const std::array<gsl::span<const o2::trk::ROFRecord>, nLayers>& layerROFs);

  /// TF anchor IR: the earliest first-ROF BCData seen across all layers when
  /// deriveAndInitTiming() was called. All LayerTiming BC values (and any BC
  /// the tracker emits via clockLayer.getROFStartInBC) are offsets from this
  /// anchor — add anchor.toLong() to convert back to absolute BC.
  const o2::InteractionRecord& getTFAnchorIR() const noexcept { return mTFAnchorIR; }

 private:
  /// One-shot setup of the per-layer LayerTiming and the three ROF lookup tables
  /// (overlap, vertex lookup, multiplicity mask). Idempotent: subsequent calls are
  /// no-ops, so the data-loading entry points may invoke it on every TF without
  /// rebuilding the tables. Mirrors the initOnceDone gate in
  /// ITSTrackingInterface::updateTimeDependentParams.
  void initTimingTables(const std::array<o2::its::LayerTiming, nLayers>& timings);

  bool mTimingTablesInitialised{false};
  o2::InteractionRecord mTFAnchorIR{0, 0};
};

} // namespace trk
} // namespace o2

#endif // ALICEO2_ALICE3GLOBALRECONSTRUCTION_TIMEFRAME_H
