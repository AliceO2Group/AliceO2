// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Tracker.h
/// \brief Track finding from MFT clusters
/// \author bogdan.vulpescu@cern.ch
/// \date Feb. 7, 2018

#ifndef ALICEO2_MFT_TRACKER_H_
#define ALICEO2_MFT_TRACKER_H_

#include <vector>

#include "DataFormatsMFT/TrackMFT.h"
#include "MFTBase/Constants.h"
#include "MFTBase/GeometryTGeo.h"

namespace o2
{
class MCCompLabel;

namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace itsmft
{
class Cluster;
}

namespace mft
{
class Tracker
{
  using Cluster = o2::itsmft::Cluster;

 public:
  Tracker(Int_t nThreads = 1);
  Tracker(const Tracker&) = delete;
  Tracker& operator=(const Tracker& tr) = delete;
  ~Tracker() = default;

  void setNumberOfThreads(Int_t n) { mNumOfThreads = n; }
  Int_t getNumberOfThreads() const { return mNumOfThreads; }

  // These functions must be implemented
  void process(const std::vector<Cluster>& clusters, std::vector<TrackMFT>& tracks);
  void processFrame(std::vector<TrackMFT>& tracks);
  const Cluster* getCluster(Int_t index) const;
  void setGeometry(o2::mft::GeometryTGeo* geom);
  void setMCTruthContainers(const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clsLabels,
                            o2::dataformats::MCTruthContainer<o2::MCCompLabel>* trkLabels)
  {
    mClsLabels = clsLabels;
    mTrkLabels = trkLabels;
  }

  void setContinuousMode(Bool_t mode) { mContinuousMode = mode; }
  bool getContinuousMode() { return mContinuousMode; }

  class Layer;

 protected:
  int loadClusters(const std::vector<Cluster>& clusters);
  void unloadClusters();
  std::vector<TrackMFT> trackInThread(Int_t first, Int_t last);

 private:
  Bool_t mContinuousMode = true;                                                  ///< triggered or cont. mode
  const o2::mft::GeometryTGeo* mGeom = nullptr;                                   ///< interface to geometry
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClsLabels = nullptr; ///< Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mTrkLabels = nullptr;       ///< Track MC labels
  std::uint32_t mROFrame = 0;                                                     ///< last frame processed
  Int_t mNumOfThreads;                                                            ///< Number of tracking threads
  static Layer sLayers[constants::LayersNumber];                                  ///< MFT layers of ladders
};

class Tracker::Layer
{
 public:
  Layer();
  Layer(const Layer&) = delete;
  Layer& operator=(const Layer& tr) = delete;
  void init();
  Bool_t insertCluster(const Cluster* c);
  void unloadClusters();
  const Cluster* getCluster(Int_t i) const { return mClusters[i]; }
  void setGeometry(o2::mft::GeometryTGeo* geom) { mGeom = geom; }
  Int_t getNumberOfClusters() const { return mClusters.size(); }

 protected:
  const o2::mft::GeometryTGeo* mGeom = nullptr;
  std::vector<const Cluster*> mClusters;
};
} // namespace mft
} // namespace o2

#endif
