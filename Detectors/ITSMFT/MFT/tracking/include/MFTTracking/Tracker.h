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
/// \file Tracker.h
/// \brief Class for the standalone track finding
///

#ifndef O2_MFT_TRACKER_H_
#define O2_MFT_TRACKER_H_

#include "MFTTracking/ROframe.h"
#include "MFTTracking/TrackFitter.h"
#include "MFTTracking/Cluster.h"
#include "MFTTracking/TrackerConfig.h"

#include "MathUtils/Utils.h"
#include "MathUtils/Cartesian.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonDataFormat/IRFrame.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsParameters/GRPObject.h"

namespace o2
{
namespace mft
{

using o2::dataformats::IRFrame;
using o2::itsmft::ROFRecord;
typedef std::function<bool(const ROFRecord&)> ROFFilter;

class T;

template <typename T>
class Tracker : public TrackerConfig
{

 public:
  Tracker(bool useMC);
  ~Tracker() = default;

  Tracker(const Tracker&) = delete;
  Tracker& operator=(const Tracker&) = delete;

  void setBz(Float_t bz);
  const Float_t getBz() const { return mBz; }

  auto& getTrackLabels() { return mTrackLabels; }

  void clearTracks()
  {
    mTrackLabels.clear();
  }

  void findTracks(ROframe<T>& rofData)
  {
    if (!mFullClusterScan) {
      clearSorting();
      sortClusters(rofData);
    }
    findLTFTracks(rofData);
    findCATracks(rofData);
  };

  void findLTFTracks(ROframe<T>&);
  void findCATracks(ROframe<T>&);
  bool fitTracks(ROframe<T>&);
  void computeTracksMClabels(const std::vector<T>&);

  void configure(const MFTTrackingParam& trkParam, bool firstTracker);
  void initializeFinder();

 private:
  void findTracksLTF(ROframe<T>&);
  void findTracksCA(ROframe<T>&);
  void findTracksLTFfcs(ROframe<T>&);
  void findTracksCAfcs(ROframe<T>&);
  void computeCellsInRoad(ROframe<T>&);
  void runForwardInRoad();
  void runBackwardInRoad(ROframe<T>&);
  void updateCellStatusInRoad();

  void sortClusters(ROframe<T>& rof)
  {
    Int_t nClsInLayer, binPrevIndex, clsMinIndex, clsMaxIndex, jClsLayer;
    // sort the clusters in R-Phi
    for (Int_t iLayer = 0; iLayer < constants::mft::LayersNumber; ++iLayer) {
      if (rof.getClustersInLayer(iLayer).size() == 0) {
        continue;
      }
      // sort clusters in layer according to the bin index
      sort(rof.getClustersInLayer(iLayer).begin(), rof.getClustersInLayer(iLayer).end(),
           [](Cluster& c1, Cluster& c2) { return c1.indexTableBin < c2.indexTableBin; });
      // find the cluster local index range in each bin
      // index = element position in the vector
      nClsInLayer = rof.getClustersInLayer(iLayer).size();
      binPrevIndex = rof.getClustersInLayer(iLayer).at(0).indexTableBin;
      clsMinIndex = 0;
      for (jClsLayer = 1; jClsLayer < nClsInLayer; ++jClsLayer) {
        if (rof.getClustersInLayer(iLayer).at(jClsLayer).indexTableBin == binPrevIndex) {
          continue;
        }

        clsMaxIndex = jClsLayer - 1;

        mClusterBinIndexRange[iLayer][binPrevIndex] = std::pair<Int_t, Int_t>(clsMinIndex, clsMaxIndex);

        binPrevIndex = rof.getClustersInLayer(iLayer).at(jClsLayer).indexTableBin;
        clsMinIndex = jClsLayer;
      } // clusters

      // last cluster
      clsMaxIndex = jClsLayer - 1;

      mClusterBinIndexRange[iLayer][binPrevIndex] = std::pair<Int_t, Int_t>(clsMinIndex, clsMaxIndex);
    } // layers
  }

  void clearSorting()
  {
    for (Int_t iLayer = 0; iLayer < constants::mft::LayersNumber; ++iLayer) {
      for (Int_t iBin = 0; iBin <= mRPhiBins + 1; ++iBin) {
        mClusterBinIndexRange[iLayer][iBin] = std::pair<Int_t, Int_t>(0, -1);
      }
    }
  }

  const Int_t isDiskFace(Int_t layer) const { return (layer % 2); }
  const Float_t getDistanceToSeed(const Cluster&, const Cluster&, const Cluster&) const;
  void getBinClusterRange(const ROframe<T>&, const Int_t, const Int_t, Int_t&, Int_t&) const;
  const Float_t getCellDeviation(const Cell&, const Cell&) const;
  const Bool_t getCellsConnect(const Cell&, const Cell&) const;
  void addCellToCurrentTrackCA(const Int_t, const Int_t, ROframe<T>&);
  void addCellToCurrentRoad(ROframe<T>&, const Int_t, const Int_t, const Int_t, const Int_t, Int_t&);

  Float_t mBz;
  std::vector<MCCompLabel> mTrackLabels;
  std::unique_ptr<o2::mft::TrackFitter<T>> mTrackFitter = nullptr;

  Int_t mMaxCellLevel = 0;

  bool mUseMC = false;

  /// helper to store points of a track candidate
  struct TrackElement {
    TrackElement() = default;
    TrackElement(Int_t la, Int_t id)
    {
      layer = la;
      idInLayer = id;
    };
    Int_t layer;
    Int_t idInLayer;
  };

  /// current road for CA algorithm
  Road mRoad;
};

//_________________________________________________________________________________________________
template <typename T>
inline const Float_t Tracker<T>::getDistanceToSeed(const Cluster& cluster1, const Cluster& cluster2, const Cluster& cluster) const
{
  // the seed is between "cluster1" and "cluster2" and cuts the plane
  // of the "cluster" at a distance dR from it
  Float_t dxSeed, dySeed, dzSeed, invdzSeed, dz, dR2, xSeed, ySeed;
  dxSeed = cluster2.getX() - cluster1.getX();
  dySeed = cluster2.getY() - cluster1.getY();
  dzSeed = cluster2.getZ() - cluster1.getZ();
  dz = cluster.getZ() - cluster1.getZ();
  invdzSeed = dz / dzSeed;
  xSeed = cluster1.getX() + dxSeed * invdzSeed;
  ySeed = cluster1.getY() + dySeed * invdzSeed;
  dR2 = (cluster.getX() - xSeed) * (cluster.getX() - xSeed) + (cluster.getY() - ySeed) * (cluster.getY() - ySeed);
  return dR2;
}

//_________________________________________________________________________________________________
template <typename T>
inline void Tracker<T>::getBinClusterRange(const ROframe<T>& event, const Int_t layer, const Int_t bin, Int_t& clsMinIndex, Int_t& clsMaxIndex) const
{
  const auto& pair = getClusterBinIndexRange(layer, bin);
  clsMinIndex = pair.first;
  clsMaxIndex = pair.second;
}

//_________________________________________________________________________________________________
template <typename T>
inline const Float_t Tracker<T>::getCellDeviation(const Cell& cell1, const Cell& cell2) const
{
  Float_t cell1dx = cell1.getX2() - cell1.getX1();
  Float_t cell1dy = cell1.getY2() - cell1.getY1();
  Float_t cell1dz = cell1.getZ2() - cell1.getZ1();

  Float_t cell2dx = cell2.getX2() - cell2.getX1();
  Float_t cell2dy = cell2.getY2() - cell2.getY1();
  Float_t cell2dz = cell2.getZ2() - cell2.getZ1();

  Float_t cell1mod = std::sqrt(cell1dx * cell1dx + cell1dy * cell1dy + cell1dz * cell1dz);
  Float_t cell2mod = std::sqrt(cell2dx * cell2dx + cell2dy * cell2dy + cell2dz * cell2dz);

  Float_t cosAngle = (cell1dx * cell2dx + cell1dy * cell2dy + cell1dz * cell2dz) / (cell1mod * cell2mod);

  return std::acos(cosAngle);
}

//_________________________________________________________________________________________________
template <typename T>
inline const Bool_t Tracker<T>::getCellsConnect(const Cell& cell1, const Cell& cell2) const
{
  Float_t cell1x2 = cell1.getX2();
  Float_t cell1y2 = cell1.getY2();
  Float_t cell2x1 = cell2.getX1();
  Float_t cell2y1 = cell2.getY1();
  Float_t dx = cell1x2 - cell2x1;
  Float_t dy = cell1y2 - cell2y1;
  Float_t dr2 = dx * dx + dy * dy;

  if (dr2 > (constants::mft::Resolution * constants::mft::Resolution)) {
    return kFALSE;
  }
  return kTRUE;
}

//_________________________________________________________________________________________________
template <typename T>
inline void Tracker<T>::computeTracksMClabels(const std::vector<T>& tracks)
{
  /// Moore's Voting Algorithm
  for (auto& track : tracks) {
    MCCompLabel maxOccurrencesValue{-1, -1, -1, false};
    int count{0};
    bool isFakeTrack{false};
    auto nClusters = track.getNumberOfPoints();
    for (int iCluster = 0; iCluster < nClusters; ++iCluster) {
      const MCCompLabel& currentLabel = track.getMCCompLabels()[iCluster];
      if (currentLabel == maxOccurrencesValue) {
        ++count;
      } else {
        if (count != 0) { // only in the first iteration count can be 0 at this point
          --count;
        }
        if (count == 0) {
          maxOccurrencesValue = currentLabel;
          count = 1;
        }
      }
    }
    count = 0;
    for (int iCluster = 0; iCluster < nClusters; ++iCluster) {
      if (track.getMCCompLabels()[iCluster] == maxOccurrencesValue) {
        count++;
      }
    }

    auto labelratio = 1.0 * count / nClusters;
    if (labelratio < mTrueTrackMCThreshold) {
      isFakeTrack = true;
      maxOccurrencesValue.setFakeFlag();
    }
    mTrackLabels.emplace_back(maxOccurrencesValue);
  }
}

} // namespace mft
} // namespace o2

#endif /* O2_MFT_TRACKER_H_ */
