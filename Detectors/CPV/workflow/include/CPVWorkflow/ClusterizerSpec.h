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

#include <vector>

#include "DataFormatsCPV/Cluster.h"
#include "DataFormatsCPV/Digit.h"
#include "CPVBase/Geometry.h"
#include "CPVReconstruction/Clusterer.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{

namespace cpv
{

namespace reco_workflow
{

/// \class ClusterizerSpec
/// \brief Clusterizer task for CPV digits
/// \author Dmitri Peresunko
/// \since Dec 14, 2019
///
/// Task to clusterize CPV digits into clusters
///
class ClusterizerSpec : public framework::Task
{
 public:
  /// \brief Constructor
  ClusterizerSpec(bool propagateMC) : framework::Task(), mPropagateMC(propagateMC) {}

  /// \brief Destructor
  ~ClusterizerSpec() override = default;

  /// \brief Initializing the ClusterizerSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Clusterizes digits into clusters
  /// \param ctx Processing context
  ///
  /// The following branches are linked:
  /// Input digits: {"CPV", "DIGITS", 0, Lifetime::Timeframe}
  /// Output clusters: {"CPVClu", "CLUSTERS", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;

 private:
  bool mPropagateMC = false;       ///< Switch whether to process MC true labels
  o2::cpv::Clusterer mClusterizer; ///< Clusterizer object
  std::vector<o2::cpv::Cluster> mOutputClusters;
  std::vector<o2::cpv::TriggerRecord> mOutputClusterTrigRecs;
  std::vector<o2::cpv::Digit> mCalibDigits;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mOutputTruthCont;
};

/// \brief Creating DataProcessorSpec for the CPV Clusterizer Spec
///
/// Refer to ClusterizerSpec::run for input and output specs
framework::DataProcessorSpec getClusterizerSpec(bool propagateMC);

} // namespace reco_workflow

} // namespace cpv

} // namespace o2
