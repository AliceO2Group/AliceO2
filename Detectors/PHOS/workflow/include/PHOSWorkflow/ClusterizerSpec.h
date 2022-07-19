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

#include "DataFormatsPHOS/Cluster.h"
#include "PHOSBase/Geometry.h"
#include "PHOSReconstruction/Clusterer.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{

namespace phos
{

namespace reco_workflow
{

/// \class ClusterizerSpec
/// \brief Clusterizer task for PHOS digits
/// \author Dmitri Peresunko
/// \since Dec 14, 2019
///
/// Task to clusterize PHOS digits into clusters
///
class ClusterizerSpec : public framework::Task
{
 public:
  /// \brief Constructor
  ClusterizerSpec(bool propagateMC, bool scanDigits, bool outputFullClu, bool defBadMap) : framework::Task(), mPropagateMC(propagateMC), mUseDigits(scanDigits), mFullCluOutput(outputFullClu), mDefBadMap(defBadMap) {}

  /// \brief Destructor
  ~ClusterizerSpec() override = default;

  /// \brief Initializing the ClusterizerSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Clusterizes digits into clusters
  /// \param ctx Processing context
  ///
  /// The following branches are linked:
  /// Input digits: {"PHS", "DIGITS", 0, Lifetime::Timeframe}
  /// Output clusters: {"PHSClu", "CLUSTERS", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;

 private:
  bool mPropagateMC = false;        ///< Switch whether to process MC true labels
  bool mUseDigits = false;          ///< Make clusters from digits or cells
  bool mFullCluOutput = false;      ///< Write full of reduced (no contributed digits) clusters
  bool mHasCalib = false;           ///< Were calibration objects received
  bool mDefBadMap = false;          ///< Use default bad map and calibration or extract from CCDB
  bool mInitSimParams = true;       ///< read sim params
  o2::phos::Clusterer mClusterizer; ///< Clusterizer object
  std::unique_ptr<CalibParams> mCalibParams;
  std::unique_ptr<BadChannelsMap> mBadMap;
  std::vector<o2::phos::Cluster> mOutputClusters;
  std::vector<o2::phos::CluElement> mOutputCluElements;
  std::vector<o2::phos::TriggerRecord> mOutputClusterTrigRecs;
  o2::dataformats::MCTruthContainer<o2::phos::MCLabel> mOutputTruthCont;
};

/// \brief Creating DataProcessorSpec for the PHOS Clusterizer Spec
///
/// Refer to ClusterizerSpec::run for input and output specs
framework::DataProcessorSpec getClusterizerSpec(bool propagateMC, bool fillFullClu, bool defBadMap = false);
framework::DataProcessorSpec getCellClusterizerSpec(bool propagateMC, bool fillFullClu, bool defBadMap = false);

} // namespace reco_workflow

} // namespace phos

} // namespace o2
