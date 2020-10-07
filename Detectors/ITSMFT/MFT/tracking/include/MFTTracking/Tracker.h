// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "MathUtils/Utils.h"
#include "MathUtils/Cartesian.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsParameters/GRPObject.h"

namespace o2
{
namespace mft
{

class TrackLTF;

class Tracker
{

 public:
  Tracker(bool useMC);
  ~Tracker() = default;

  Tracker(const Tracker&) = delete;
  Tracker& operator=(const Tracker&) = delete;

  void setBz(Float_t bz);
  const Float_t getBz() const { return mBz; }

  std::vector<TrackMFTExt>& getTracks() { return mTracks; }
  std::vector<TrackLTF>& getTracksLTF() { return mTracksLTF; }
  o2::dataformats::MCTruthContainer<MCCompLabel>& getTrackLabels() { return mTrackLabels; }

  void clustersToTracks(ROframe&, std::ostream& = std::cout);

  template <class T>
  void computeTracksMClabels(const T&);

  void setROFrame(std::uint32_t f) { mROFrame = f; }
  std::uint32_t getROFrame() const { return mROFrame; }

 private:
  void findTracks(ROframe&);
  void findTracksLTF(ROframe&);
  void findTracksCA(ROframe&);
  void computeCells(ROframe&);
  void computeCellsInRoad(Road&);
  void runForwardInRoad(ROframe&);
  void runBackwardInRoad(ROframe&);
  void updateCellStatusInRoad(Road&);

  bool fitTracks(ROframe&);

  const Int_t isDiskFace(Int_t layer) const { return (layer % 2); }
  const Float_t getDistanceToSeed(const Cluster&, const Cluster&, const Cluster&) const;
  void getRPhiProjectionBin(const Cluster&, const Int_t, const Int_t, Int_t&, Int_t&) const;
  Bool_t getBinClusterRange(const ROframe&, const Int_t, const Int_t, Int_t&, Int_t&) const;
  const Float_t getCellDeviation(const ROframe&, const Cell&, const Cell&) const;
  const Bool_t getCellsConnect(const ROframe&, const Cell&, const Cell&) const;
  const Float_t getCellChisquare(ROframe&, const Cell&) const;
  const Bool_t addCellToCurrentTrackCA(const Int_t, const Int_t, ROframe&);

  const Bool_t LinearRegression(Int_t, Float_t*, Float_t*, Float_t*, Float_t&, Float_t&, Float_t&, Float_t&, Float_t&, Int_t skip = -1) const;

  Float_t mBz = 5.f;
  std::uint32_t mROFrame = 0;
  std::vector<TrackMFTExt> mTracks;
  std::vector<TrackLTF> mTracksLTF;
  std::vector<Cluster> mClusters;
  o2::dataformats::MCTruthContainer<MCCompLabel> mTrackLabels;
  std::unique_ptr<o2::mft::TrackFitter> mTrackFitter = nullptr;

  Int_t mMaxCellLevel = 0;

  bool mUseMC = false;
};

//_________________________________________________________________________________________________
inline const Float_t Tracker::getDistanceToSeed(const Cluster& cluster1, const Cluster& cluster2, const Cluster& cluster) const
{
  // the seed is between "cluster1" and "cluster2" and cuts the plane
  // of the "cluster" at a distance dR from it
  Float_t dxSeed, dySeed, dzSeed, dz, dR, xSeed, ySeed;
  dxSeed = cluster2.getX() - cluster1.getX();
  dySeed = cluster2.getY() - cluster1.getY();
  dzSeed = cluster2.getZ() - cluster1.getZ();
  dz = cluster.getZ() - cluster1.getZ();
  xSeed = cluster1.getX() + dxSeed * dz / dzSeed;
  ySeed = cluster1.getY() + dySeed * dz / dzSeed;
  dR = std::sqrt((cluster.getX() - xSeed) * (cluster.getX() - xSeed) + (cluster.getY() - ySeed) * (cluster.getY() - ySeed));
  return dR;
}

//_________________________________________________________________________________________________
inline void Tracker::getRPhiProjectionBin(const Cluster& cluster1, const Int_t layer1, const Int_t layer, Int_t& binR_proj, Int_t& binPhi_proj) const
{
  Float_t dz, x_proj, y_proj, r_proj, phi_proj;
  dz = constants::mft::LayerZCoordinate()[layer] - constants::mft::LayerZCoordinate()[layer1];
  x_proj = cluster1.getX() + dz * cluster1.getX() * constants::mft::InverseLayerZCoordinate()[layer1];
  y_proj = cluster1.getY() + dz * cluster1.getY() * constants::mft::InverseLayerZCoordinate()[layer1];
  auto clsPoint2D = Point2D<Float_t>(x_proj, y_proj);
  r_proj = clsPoint2D.R();
  phi_proj = clsPoint2D.Phi();
  o2::utils::BringTo02PiGen(phi_proj);
  binR_proj = constants::index_table::getRBinIndex(r_proj);
  binPhi_proj = constants::index_table::getPhiBinIndex(phi_proj);
  return;
}

//_________________________________________________________________________________________________
inline Bool_t Tracker::getBinClusterRange(const ROframe& event, const Int_t layer, const Int_t bin, Int_t& clsMinIndex, Int_t& clsMaxIndex) const
{
  const auto pair2 = event.getClusterBinIndexRange(layer).find(bin);
  if (pair2 == event.getClusterBinIndexRange(layer).end())
    return kFALSE;
  Int_t binIndex = pair2->first;
  // get the range in ordered cluster index within this bin
  std::pair<Int_t, Int_t> pair1 = pair2->second;
  clsMinIndex = pair1.first;
  clsMaxIndex = pair1.second;
  return kTRUE;
}

//_________________________________________________________________________________________________
inline const Float_t Tracker::getCellDeviation(const ROframe& event, const Cell& cell1, const Cell& cell2) const
{
  Int_t cell1layer1 = cell1.getFirstLayerId();
  Int_t cell1layer2 = cell1.getSecondLayerId();

  Int_t cell2layer1 = cell2.getFirstLayerId();
  Int_t cell2layer2 = cell2.getSecondLayerId();

  Int_t cell1cls1 = cell1.getFirstClusterIndex();
  Int_t cell1cls2 = cell1.getSecondClusterIndex();

  Int_t cell2cls1 = cell2.getFirstClusterIndex();
  Int_t cell2cls2 = cell2.getSecondClusterIndex();

  auto cluster11 = event.getClustersInLayer(cell1layer1)[cell1cls1];
  auto cluster12 = event.getClustersInLayer(cell1layer2)[cell1cls2];
  auto cluster21 = event.getClustersInLayer(cell2layer1)[cell2cls1];
  auto cluster22 = event.getClustersInLayer(cell2layer2)[cell2cls2];

  Float_t cell1x1 = cluster11.getX();
  Float_t cell1y1 = cluster11.getY();
  Float_t cell1z1 = cluster11.getZ();

  Float_t cell1x2 = cluster12.getX();
  Float_t cell1y2 = cluster12.getY();
  Float_t cell1z2 = cluster12.getZ();

  Float_t cell2x1 = cluster21.getX();
  Float_t cell2y1 = cluster21.getY();
  Float_t cell2z1 = cluster21.getZ();

  Float_t cell2x2 = cluster22.getX();
  Float_t cell2y2 = cluster22.getY();
  Float_t cell2z2 = cluster22.getZ();

  Float_t cell1dx = cell1x2 - cell1x1;
  Float_t cell1dy = cell1y2 - cell1y1;
  Float_t cell1dz = cell1z2 - cell1z1;

  Float_t cell2dx = cell2x2 - cell2x1;
  Float_t cell2dy = cell2y2 - cell2y1;
  Float_t cell2dz = cell2z2 - cell2z1;

  Float_t cell1mod = std::sqrt(cell1dx * cell1dx + cell1dy * cell1dy + cell1dz * cell1dz);
  Float_t cell2mod = std::sqrt(cell2dx * cell2dx + cell2dy * cell2dy + cell2dz * cell2dz);

  Float_t cosAngle = (cell1dx * cell2dx + cell1dy * cell2dy + cell1dz * cell2dz) / (cell1mod * cell2mod);

  return std::acos(cosAngle);
}

//_________________________________________________________________________________________________
inline const Bool_t Tracker::getCellsConnect(const ROframe& event, const Cell& cell1, const Cell& cell2) const
{
  Int_t cell1layer1 = cell1.getFirstLayerId();
  Int_t cell1layer2 = cell1.getSecondLayerId();

  Int_t cell2layer1 = cell2.getFirstLayerId();
  Int_t cell2layer2 = cell2.getSecondLayerId();

  Int_t cell1cls1 = cell1.getFirstClusterIndex();
  Int_t cell1cls2 = cell1.getSecondClusterIndex();

  Int_t cell2cls1 = cell2.getFirstClusterIndex();
  Int_t cell2cls2 = cell2.getSecondClusterIndex();

  auto cluster11 = event.getClustersInLayer(cell1layer1)[cell1cls1];
  auto cluster12 = event.getClustersInLayer(cell1layer2)[cell1cls2];
  auto cluster21 = event.getClustersInLayer(cell2layer1)[cell2cls1];
  auto cluster22 = event.getClustersInLayer(cell2layer2)[cell2cls2];

  Float_t cell1x1 = cluster11.getX();
  Float_t cell1y1 = cluster11.getY();
  //Float_t cell1z1 = cluster11.getZ();

  Float_t cell1x2 = cluster12.getX();
  Float_t cell1y2 = cluster12.getY();
  //Float_t cell1z2 = cluster12.getZ();

  Float_t cell2x1 = cluster21.getX();
  Float_t cell2y1 = cluster21.getY();
  //Float_t cell2z1 = cluster21.getZ();

  Float_t cell2x2 = cluster22.getX();
  Float_t cell2y2 = cluster22.getY();
  //Float_t cell2z2 = cluster22.getZ();

  Float_t cell1dx = cell1x2 - cell1x1;
  Float_t cell1dy = cell1y2 - cell1y1;
  //Float_t cell1dz = cell1z2 - cell1z1;

  Float_t cell2dx = cell2x2 - cell2x1;
  Float_t cell2dy = cell2y2 - cell2y1;
  //Float_t cell2dz = cell2z2 - cell2z1;

  Float_t dx = cell1x2 - cell2x1;
  Float_t dy = cell1y2 - cell2y1;
  Float_t dr = std::sqrt(dx * dx + dy * dy);

  if (dr > constants::mft::Resolution) {
    return kFALSE;
  }
  return kTRUE;
}

//_________________________________________________________________________________________________
template <class T>
inline void Tracker::computeTracksMClabels(const T& tracks)
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
      if (track.getMCCompLabels()[iCluster] == maxOccurrencesValue)
        count++;
    }

    auto labelratio = 1.0 * count / nClusters;
    if (labelratio >= 0.8) {
    } else {
      isFakeTrack = true;
      maxOccurrencesValue.setFakeFlag();
    }
    mTrackLabels.addElement(mTrackLabels.getIndexedSize(), maxOccurrencesValue);
  }
}

} // namespace mft
} // namespace o2

#endif /* O2_MFT_TRACKER_H_ */
