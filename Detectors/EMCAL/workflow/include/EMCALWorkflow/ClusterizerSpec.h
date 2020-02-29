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
#include "EMCALBase/Geometry.h"
#include "EMCALReconstruction/Clusterizer.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{

namespace emcal
{

namespace reco_workflow
{

/// \class ClusterizerSpec
/// \brief Clusterizer task for EMCAL digits
/// \ingroup EMCALworkflow
/// \author Ruediger Haake  <ruediger.haake@cern.ch>, Yale University
/// \since Oct 23, 2019
///
/// Task to clusterize EMCAL digits into clusters
/// The resulting cluster objects contain a range of digits
/// that can be found in output digit indices object
///
template <class InputType>
class ClusterizerSpec : public framework::Task
{
 public:
  /// \brief Constructor
  ClusterizerSpec() : framework::Task(){};

  /// \brief Destructor
  ~ClusterizerSpec() override = default;

  /// \brief Initializing the ClusterizerSpec
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Run conversion of digits to cells
  /// \param ctx Processing context
  ///
  /// Clusterizes digits into clusters
  ///
  /// The following branches are linked:
  /// Input digits: {"EMC", "DIGITS", 0, Lifetime::Timeframe}
  /// Output clusters: {"clusters", "CLUSTERS", 0, Lifetime::Timeframe}
  /// Output indices: {"clusterDigitIndices", "CLUSTERDIGITINDICES", 0, Lifetime::Timeframe}
  void run(framework::ProcessingContext& ctx) final;

 private:
  o2::emcal::Clusterizer<InputType> mClusterizer;                                ///< Clusterizer object
  o2::emcal::Geometry* mGeometry = nullptr;                                      ///< Pointer to geometry object
  const std::vector<o2::emcal::Cluster>* mOutputClusters = nullptr;              ///< Container with output clusters (pointer)
  const std::vector<o2::emcal::ClusterIndex>* mOutputCellDigitIndices = nullptr; ///< Container with indices of cluster digits (pointer)
};

/// \brief Creating DataProcessorSpec for the EMCAL Clusterizer Spec
/// \ingroup EMCALworkflow
///
/// Refer to ClusterizerSpec::run for input and output specs
framework::DataProcessorSpec getClusterizerSpec(bool useDigits);

} // namespace reco_workflow

} // namespace emcal

} // namespace o2
