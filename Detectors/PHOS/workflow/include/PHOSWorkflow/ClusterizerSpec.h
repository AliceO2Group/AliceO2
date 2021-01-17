// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  ClusterizerSpec(bool propagateMC, bool scanDigits) : framework::Task(), mPropagateMC(propagateMC), mUseDigits(scanDigits) {}

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
  bool mPropagateMC = false; ///< Switch whether to process MC true labels
  bool mUseDigits = false;
  o2::phos::Clusterer mClusterizer; ///< Clusterizer object
  std::vector<o2::phos::Cluster> mOutputClusters;
  std::vector<o2::phos::TriggerRecord> mOutputClusterTrigRecs;
  o2::dataformats::MCTruthContainer<o2::phos::MCLabel> mOutputTruthCont;
};

/// \brief Creating DataProcessorSpec for the PHOS Clusterizer Spec
///
/// Refer to ClusterizerSpec::run for input and output specs
framework::DataProcessorSpec getClusterizerSpec(bool propagateMC);
framework::DataProcessorSpec getCellClusterizerSpec(bool propagateMC);

} // namespace reco_workflow

} // namespace phos

} // namespace o2
