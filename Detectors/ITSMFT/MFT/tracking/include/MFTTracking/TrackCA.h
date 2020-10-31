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
/// \file TrackCA.h
/// \brief Standalone classes for the track found by the Linear-Track-Finder (LTF) and by the Cellular-Automaton (CA)
///

#ifndef O2_MFT_TRACKCA_H_
#define O2_MFT_TRACKCA_H_

#include <array>
#include "DataFormatsMFT/TrackMFT.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "MFTTracking/Constants.h"
#include "MFTTracking/Cluster.h"
#include <fairlogger/Logger.h>

namespace o2
{
namespace mft
{

class TrackLTF : public TrackMFTExt
{
 public:
  TrackLTF() = default;
  TrackLTF(const TrackLTF& t) = default;
  ~TrackLTF() = default;
  const std::array<Float_t, constants::mft::LayersNumber>& getXCoordinates() const { return mX; }
  const std::array<Float_t, constants::mft::LayersNumber>& getYCoordinates() const { return mY; }
  const std::array<Float_t, constants::mft::LayersNumber>& getZCoordinates() const { return mZ; }
  const std::array<Float_t, constants::mft::LayersNumber>& getSigmasX2() const { return mSigmaX2; }
  const std::array<Float_t, constants::mft::LayersNumber>& getSigmasY2() const { return mSigmaY2; }
  const std::array<Int_t, constants::mft::LayersNumber>& getLayers() const { return mLayer; }
  const std::array<Int_t, constants::mft::LayersNumber>& getClustersId() const { return mClusterId; }
  const std::array<MCCompLabel, constants::mft::LayersNumber>& getMCCompLabels() const { return mMCCompLabels; }
  void setPoint(const Cluster, const Int_t layer, const Int_t clusterId, const MCCompLabel label, const Int_t extClsIndex, Bool_t& newPoint);

  void sort();

 private:
  std::array<Float_t, constants::mft::LayersNumber> mX = {-25., -25., -25., -25., -25., -25., -25., -25., -25., -25.};
  std::array<Float_t, constants::mft::LayersNumber> mY = {-25., -25., -25., -25., -25., -25., -25., -25., -25., -25.};
  std::array<Float_t, constants::mft::LayersNumber> mZ = {-120., -120., -120., -120., -120., -120., -120., -120., -120., -120.};
  std::array<Float_t, constants::mft::LayersNumber> mSigmaX2 = {0};
  std::array<Float_t, constants::mft::LayersNumber> mSigmaY2 = {0};
  std::array<Int_t, constants::mft::LayersNumber> mLayer;
  std::array<Int_t, constants::mft::LayersNumber> mClusterId;
  std::array<MCCompLabel, constants::mft::LayersNumber> mMCCompLabels;
};

//_________________________________________________________________________________________________
class TrackCA : public TrackLTF
{
 public:
  TrackCA() = default;
  TrackCA(const TrackCA& t) = default;
  ~TrackCA() = default;
  void addCell(const Int_t, const Int_t);
  void removeLastCell(Int_t&, Int_t&);
  const Int_t getNCells() const { return mNCells; }
  void setRoadId(const Int_t rid) { mRoadId = rid; }
  const Int_t getRoadId() const { return mRoadId; }
  void setChiSquareZX(const Float_t chisq) { mChiSquareZX = chisq; }
  void setChiSquareZY(const Float_t chisq) { mChiSquareZY = chisq; }
  const Float_t getChiSquareZX() const { return mChiSquareZX; }
  const Float_t getChiSquareZY() const { return mChiSquareZY; }

  const std::array<Int_t, constants::mft::LayersNumber>& getCellsLayer() const { return mCellLayer; }
  const std::array<Int_t, constants::mft::LayersNumber>& getCellsId() const { return mCellId; }

 private:
  Int_t mNCells{0};
  Int_t mRoadId{-1};
  Float_t mChiSquareZX{0.};
  Float_t mChiSquareZY{0.};

  std::array<Int_t, constants::mft::LayersNumber> mCellLayer;
  std::array<Int_t, constants::mft::LayersNumber> mCellId;
  ClassDefNV(TrackCA, 2);
};

//_________________________________________________________________________________________________
inline void TrackCA::addCell(const Int_t layer, const Int_t cellId)
{
  mCellLayer[mNCells] = layer;
  mCellId[mNCells] = cellId;
  mNCells++;
}

//_________________________________________________________________________________________________
inline void TrackCA::removeLastCell(Int_t& layer, Int_t& cellId)
{
  layer = mCellLayer[mNCells - 1];
  cellId = mCellId[mNCells - 1];

  auto nPoints = getNumberOfPoints();
  if (nPoints == 2) { // we have only a single cell in the track
    setNumberOfPoints(--nPoints);
  }
  setNumberOfPoints(--nPoints);
  mNCells--;
}

//_________________________________________________________________________________________________
inline void TrackLTF::setPoint(const Cluster cl, const Int_t layer, const Int_t clusterId, const MCCompLabel label, const Int_t extClsIndex, Bool_t& newPoint)
{
  auto nPoints = getNumberOfPoints();
  if (newPoint) {
    if (nPoints > 0) {
      if (mZ[nPoints - 1] == cl.getZ()) {
        LOG(WARN) << "MFT TrackLTF: skipping setPoint (1 cluster per layer!)";
        return;
      }
    }
    if (nPoints > constants::mft::LayersNumber) {
      LOG(WARN) << "MFT TrackLTF Overflow";
      return;
    }
    mX[nPoints] = cl.getX();
    mY[nPoints] = cl.getY();
    mZ[nPoints] = cl.getZ();
    mSigmaX2[nPoints] = cl.sigmaX2;
    mSigmaY2[nPoints] = cl.sigmaY2;
    mLayer[nPoints] = layer;
    mClusterId[nPoints] = clusterId;
    mMCCompLabels[nPoints] = label;
    setExternalClusterIndex(nPoints, extClsIndex);
    setNumberOfPoints(nPoints + 1);
  } else {
    mX[nPoints] = cl.getX();
    mY[nPoints] = cl.getY();
    mZ[nPoints] = cl.getZ();
    mSigmaX2[nPoints] = cl.sigmaX2;
    mSigmaY2[nPoints] = cl.sigmaY2;
    mLayer[nPoints] = layer;
    mClusterId[nPoints] = clusterId;
    mMCCompLabels[nPoints] = label;
    setExternalClusterIndex(nPoints, extClsIndex);
  }
}

//_________________________________________________________________________________________________
inline void TrackLTF::sort()
{
  // Orders elements along z position
  struct ClusterData {
    Float_t x;
    Float_t y;
    Float_t z;
    Float_t sigmaX2;
    Float_t sigmaY2;
    Int_t layer;
    Int_t clusterId;
    MCCompLabel label;
    Int_t extClsIndex;
  };
  std::vector<ClusterData> points;

  // Loading cluster data
  for (Int_t point = 0; point < getNumberOfPoints(); ++point) {
    auto& somepoint = points.emplace_back();
    somepoint.x = mX[point];
    somepoint.y = mY[point];
    somepoint.z = mZ[point];
    somepoint.sigmaX2 = mSigmaX2[point];
    somepoint.sigmaY2 = mSigmaY2[point];
    somepoint.layer = mLayer[point];
    somepoint.clusterId = mClusterId[point];
    somepoint.label = mMCCompLabels[point];
    somepoint.extClsIndex = mExtClsIndex[point];
  }

  // Sorting cluster data
  std::sort(points.begin(), points.end(), [](ClusterData a, ClusterData b) { return a.z > b.z; });

  // Storing sorted cluster data
  for (Int_t point = 0; point < getNumberOfPoints(); ++point) {
    mX[point] = points[point].x;
    mY[point] = points[point].y;
    mZ[point] = points[point].z;
    mSigmaX2[point] = points[point].sigmaX2;
    mSigmaY2[point] = points[point].sigmaY2;
    mLayer[point] = points[point].layer;
    mClusterId[point] = points[point].clusterId;
    mMCCompLabels[point] = points[point].label;
    mExtClsIndex[point] = points[point].extClsIndex;
  }
}

} // namespace mft

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::mft::TrackCA> : std::true_type {
};

template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::mft::TrackLTF> : std::true_type {
};

} // namespace framework
} // namespace o2
#endif /* O2_MFT_TRACKCA_H_ */
