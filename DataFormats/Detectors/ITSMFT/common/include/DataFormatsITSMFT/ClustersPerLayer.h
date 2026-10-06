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

/// \file ClustersPerLayer.h
/// \brief Container of the ITS/MFT clusters addressed by the composed (layer,index) ID
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_ITSMFT_CLUSTERSPERLAYER_H
#define ALICEO2_ITSMFT_CLUSTERSPERLAYER_H

#include "DataFormatsITSMFT/ClusterID.h"
#include "Framework/Logger.h"
#include <array>
#include <vector>

namespace o2::itsmft
{

///< Container of the ITS/MFT clusters supplied either as a single (monolithic) array or per layer
///< (staggered readout). The clusters of all layers are kept in a single vector, layer by layer,
///< with the per-layer starting offsets recorded, so that a cluster referred to by the composed ID
///< (layer << ClusLayerShift) + index_in_layer can be looked up directly. With the monolithic input
///< only the layer slot 0 is filled and the composed ID coincides with the flat cluster index, hence
///< the same lookup works for both cases.
///<
///< The conversion of the compact clusters to the stored objects is detector specific (and needs the
///< geometry), so it is left to the caller: the container only records where each layer starts.
///< Expected usage (nLr == 1 for the monolithic input):
///<   cont.init(nLr);
///<   for (int lr = 0; lr < nLr; lr++) {
///<     cont.beginLayer(lr);
///<     auto pattIt = recoData.getITSClustersPatterns(lr).begin();
///<     o2::its::ioutils::convertCompactClusters(recoData.getITSClusters(lr), pattIt, cont.getClusters(), dict);
///<   }
///<   cont.finalize();
///< after which cont[composedID] gives the cluster referred to by a track cluster reference.
template <typename T>
class ClustersPerLayer
{
 public:
  ///< prepare for filling nLr layer slots, discarding the previous content
  void init(int nLr)
  {
    if (nLr < 1 || nLr > MaxClusLayers) {
      LOGP(fatal, "Clusters container cannot be initialized for {} layers, must be within 1:{}", nLr, MaxClusLayers);
    }
    mClusters.clear();
    mLrFirst.fill(0);
    mNLayers = nLr;
  }

  ///< record the start of the layer lr data, the layers must be filled in the increasing order
  void beginLayer(int lr)
  {
    if (lr < 0 || lr >= mNLayers) {
      LOGP(fatal, "Clusters container was initialized for {} layers, cannot fill the layer {}", mNLayers, lr);
    }
    mLrFirst[lr] = int(mClusters.size());
  }

  ///< to be called once all the layers were filled
  void finalize()
  {
    for (int lr = mNLayers; lr <= MaxClusLayers; lr++) { // the slots above the filled ones are empty
      mLrFirst[lr] = int(mClusters.size());
    }
  }

  void clear()
  {
    mClusters.clear();
    mLrFirst.fill(0);
    mNLayers = 1;
  }

  ///< the clusters of all layers, to be appended to by the caller between beginLayer and finalize
  auto& getClusters() { return mClusters; }
  const auto& getClusters() const { return mClusters; }

  auto getNLayers() const { return mNLayers; }
  auto getFirstIndex(int lr) const { return mLrFirst[lr]; }
  auto getNClusters(int lr) const { return mLrFirst[lr + 1] - mLrFirst[lr]; }
  auto size() const { return mClusters.size(); }
  bool empty() const { return mClusters.empty(); }

  ///< flat index of the cluster referred to by its composed ID
  int flatIndex(int composedID) const
  {
    return mLrFirst[clusID2Layer(composedID)] + clusID2Index(composedID);
  }
  const T& operator[](int composedID) const { return mClusters[flatIndex(composedID)]; }

 private:
  std::vector<T> mClusters{};                    ///< clusters of all layers, layer by layer
  std::array<int, MaxClusLayers + 1> mLrFirst{}; ///< 1st cluster of every layer, + the total in the last slot
  int mNLayers = 1;                              ///< number of filled layer slots
};

} // namespace o2::itsmft

#endif
