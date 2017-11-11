// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_TRACKTPC
#define ALICEO2_TPC_TRACKTPC

#include "DetectorsBase/Track.h"

#include "TPCBase/Defs.h"
#include "TPCReconstruction/Cluster.h"


namespace o2 {
namespace TPC {

/// \class TrackTPC
/// This is the definition of the TPC Track Object


class TrackTPC :public o2::Base::Track::TrackParCov {
  public:

    using o2::Base::Track::TrackParCov::TrackParCov; // inherit

    /// Default constructor
    TrackTPC() = default;

    /// Destructor
    ~TrackTPC() = default;

    /// Add a single cluster to the track
    void addCluster(const Cluster &c);

    /// Add an array/vector of clusters to the track; ClusterType needs to inherit from o2::TPC::Cluster
    template <typename ClusterType>
    void addClusterArray(std::vector<ClusterType> *arr);

    /// Get the clusters which are associated with the track
    /// \return clusters of the track as a std::vector
    void getClusterVector(std::vector<Cluster> &clVec)  const { clVec = mClusterVector; }

    /// Get the truncated mean energy loss of the track
    /// \param low low end of truncation
    /// \param high high end of truncation
    /// \param type 0 for Qmax, 1 for Q
    /// \param removeRows option to remove certain rows from the dEdx calculation
    /// \param nclPID pass any pointer to have the number of used clusters written to it
    /// \return mean energy loss
    float getTruncatedMean(float low=0.05, float high=0.7, int type=1, int removeRows=0, int *nclPID=nullptr) const;

  private:
    std::vector<Cluster> mClusterVector;

};

inline
void TrackTPC::addCluster(const Cluster &c)
{
  mClusterVector.push_back(c);
}

template<typename ClusterType>
inline
void TrackTPC::addClusterArray(std::vector<ClusterType> *arr)
{
  static_assert(std::is_base_of<o2::TPC::Cluster, ClusterType>::value, "ClusterType needs to inherit from o2::TPC::Cluster");
  for (auto clusterObject : *arr){
    addCluster(clusterObject);
  }
}

}
}

#endif
