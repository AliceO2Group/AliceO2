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

#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "DataFormatsEMCAL/AnalysisCluster.h"
#include "DataFormatsEMCAL/EventHandler.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/ClusterFactory.h"
#include "EMCALReconstruction/Clusterizer.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{

namespace emcal
{

namespace reco_workflow
{

/// \class AnalysisClusterSpec
/// \brief Analysis Cluster task for EMCAL anlaysis clusters
/// \ingroup EMCALworkflow
/// \author Hadi Hassan, hadi.hassan@cern.ch ORNL
/// \since March 17, 2020
///
/// Task for testing the event builder and the cluster factory
///
template <class InputType>
class AnalysisClusterSpec : public framework::Task
{
 public:
  /// \brief Constructor
  AnalysisClusterSpec() : framework::Task(){};

  /// \brief Destructor
  ~AnalysisClusterSpec() override = default;

  /// \brief Initializing the AnalysisClusterSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Run conversion of digits to cells
  /// \param ctx Processing context
  ///
  /// Clusterizes digits into clusters
  ///
  /// The following branches are linked:
  /// Output analysis clusters: {"analysisclusters", "ANALYSISCLUSTERS", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;

 private:
  o2::emcal::Clusterizer<InputType> mClusterizer;                        ///< Clusterizer object
  o2::emcal::Geometry* mGeometry = nullptr;                              ///< Pointer to geometry object
  o2::emcal::EventHandler<InputType>* mEventHandler = nullptr;           ///< Pointer to the event builder
  o2::emcal::ClusterFactory<InputType>* mClusterFactory = nullptr;       ///< Pointer to the cluster builder
  std::vector<o2::emcal::AnalysisCluster>* mOutputAnaClusters = nullptr; ///< Container with output clusters (pointer)
};

/// \brief Creating DataProcessorSpec for the EMCAL Analysis Cluster Spec
/// \ingroup EMCALworkflow
///
/// Refer to AnalysisClusterSpec::run for input and output specs
framework::DataProcessorSpec getAnalysisClusterSpec(bool useDigits);

} // namespace reco_workflow

} // namespace emcal

} // namespace o2
