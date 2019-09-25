// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterContainer.h
/// \brief Container class for TPC clusters
#ifndef _ALICEO2_TPC_ClusterContainer_
#define _ALICEO2_TPC_ClusterContainer_

#include <vector>
#include <cassert>
#include <Rtypes.h> // for Float_t etc

namespace o2
{
namespace tpc
{

/// \class ClusterContainer
/// \brief Container class for TPC clusters
class ClusterContainer
{
 public:
  // Initialize the clones array
  // @param clusterType Possibility to store different types of clusters
  // void InitArray(const Char_t* clusterType="o2::tpc::Cluster");

  /// Add cluster to array
  /// @param output, the vector to append to
  /// @param cru CRU (sector)
  /// @param row Row
  /// @param q Total charge of cluster
  /// @param qmax Maximum charge in a single cell (pad, time)
  /// @param padmean Mean position of cluster in pad direction
  /// @param padsigma Sigma of cluster in pad direction
  /// @param timemean Mean position of cluster in time direction
  /// @param timesigma Sigma of cluster in time direction
  template <typename ClusterType>
  static ClusterType* addCluster(std::vector<ClusterType>* output,
                                 Int_t cru, Int_t row, Float_t qTot, Float_t qMax,
                                 Float_t meanpad, Float_t meantime, Float_t sigmapad,
                                 Float_t sigmatime)
  {
    assert(output);
    output->emplace_back(); // emplace_back a defaut constructed cluster of type ClusterType
    auto& cluster = output->back();
    // set its concrete parameters:
    // ATTENTION: the order of parameters in setParameters is different than in AddCluster!
    cluster.setParameters(cru, row, qTot, qMax,
                          meanpad, sigmapad,
                          meantime, sigmatime);
    return &cluster;
  }
};
} // namespace tpc
} // namespace o2

#endif
