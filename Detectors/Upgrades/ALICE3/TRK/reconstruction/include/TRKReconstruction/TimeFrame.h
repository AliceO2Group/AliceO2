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

#ifndef ALICEO2_TRK_TIMEFRAME_H
#define ALICEO2_TRK_TIMEFRAME_H

#include "ITStracking/TimeFrame.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "SimulationDataFormat/MCCompLabel.h"
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
class Hit;
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

  /// Add primary vertices from MC headers for each ROF
  /// \param mcHeaderTree Tree containing MC event headers
  /// \param nRofs Number of ROFs (Read-Out Frames)
  /// \param nEvents Number of events to process
  /// \param inROFpileup Number of events per ROF
  void getPrimaryVerticesFromMC(TTree* mcHeaderTree, int nRofs, Long64_t nEvents, int inROFpileup);
};

} // namespace trk
} // namespace o2

#endif // ALICEO2_TRK_TIMEFRAME_H
