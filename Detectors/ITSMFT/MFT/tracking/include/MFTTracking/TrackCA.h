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
#include "SimulationDataFormat/MCCompLabel.h"
#include "MFTTracking/Constants.h"
#include <fairlogger/Logger.h>

namespace o2
{
namespace mft
{

class TrackCA final
{
 public:
  TrackCA() = default;
  ~TrackCA() = default;
  void addCell(const Int_t, const Int_t);
  void removeLastCell(Int_t&, Int_t&);
  const Int_t getNCells() const;
  void setRoadId(const Int_t rid) { mRoadId = rid; }
  const Int_t getRoadId() const;
  void setPoint(const Float_t x, const Float_t y, const Float_t z, const Int_t layer, const Int_t clusterId, const MCCompLabel label, Bool_t& newPoint);
  const Int_t getNPoints() const { return mNPoints; }
  void setChiSquareZX(const Float_t chisq) { mChiSquareZX = chisq; }
  void setChiSquareZY(const Float_t chisq) { mChiSquareZY = chisq; }
  const Float_t getChiSquareZX() const { return mChiSquareZX; }
  const Float_t getChiSquareZY() const { return mChiSquareZY; }
  const std::array<Float_t, constants::mft::LayersNumber>& getXCoordinates() const { return mX; }
  const std::array<Float_t, constants::mft::LayersNumber>& getYCoordinates() const { return mY; }
  const std::array<Float_t, constants::mft::LayersNumber>& getZCoordinates() const { return mZ; }
  const std::array<Int_t, constants::mft::LayersNumber>& getLayers() const { return mLayer; }
  const std::array<Int_t, constants::mft::LayersNumber>& getClustersId() const { return mClusterId; }
  const std::array<Int_t, constants::mft::LayersNumber>& getCellsLayer() const { return mCellLayer; }
  const std::array<Int_t, constants::mft::LayersNumber>& getCellsId() const { return mCellId; }
  const std::array<MCCompLabel, constants::mft::LayersNumber>& getMCCompLabels() const { return mMCCompLabels; }

 private:
  Int_t mNPoints{0};
  Int_t mNCells{0};
  Int_t mRoadId{-1};
  Float_t mChiSquareZX{0.};
  Float_t mChiSquareZY{0.};
  std::array<Float_t, constants::mft::LayersNumber> mX = {-15., -15., -15., -15., -15., -15., -15., -15., -15., -15.};
  std::array<Float_t, constants::mft::LayersNumber> mY = {-15., -15., -15., -15., -15., -15., -15., -15., -15., -15.};
  std::array<Float_t, constants::mft::LayersNumber> mZ = {-80., -80., -80., -80., -80., -80., -80., -80., -80., -80.};
  std::array<Int_t, constants::mft::LayersNumber> mLayer;
  std::array<Int_t, constants::mft::LayersNumber> mClusterId;
  std::array<Int_t, constants::mft::LayersNumber> mCellLayer;
  std::array<Int_t, constants::mft::LayersNumber> mCellId;
  std::array<MCCompLabel, constants::mft::LayersNumber> mMCCompLabels;
  ClassDefNV(TrackCA, 1);
};

inline void TrackCA::addCell(const Int_t layer, const Int_t cellId)
{
  mCellLayer[mNCells] = layer;
  mCellId[mNCells] = cellId;
  mNCells++;
}

inline void TrackCA::removeLastCell(Int_t& layer, Int_t& cellId)
{
  layer = mCellLayer[mNCells - 1];
  cellId = mCellId[mNCells - 1];

  if (mNPoints == 2) { // we have only a single cell in the track
    mNPoints--;
  }
  mNPoints--;
  mNCells--;
}

inline const Int_t TrackCA::getNCells() const
{
  return mNCells;
}

inline const Int_t TrackCA::getRoadId() const
{
  return mRoadId;
}

inline void TrackCA::setPoint(const Float_t x, const Float_t y, const Float_t z, const Int_t layer, const Int_t clusterId, const MCCompLabel label, Bool_t& newPoint)
{
  if (newPoint) {
    if (mNPoints == constants::mft::LayersNumber) {
      LOG(WARN) << "MFT TrackLTF Overflow";
      return;
    }
    mX[mNPoints] = x;
    mY[mNPoints] = y;
    mZ[mNPoints] = z;
    mLayer[mNPoints] = layer;
    mClusterId[mNPoints] = clusterId;
    mMCCompLabels[mNPoints] = label;
    mNPoints++;
  } else {
    mX[mNPoints] = x;
    mY[mNPoints] = y;
    mZ[mNPoints] = z;
    mLayer[mNPoints] = layer;
    mClusterId[mNPoints] = clusterId;
    mMCCompLabels[mNPoints] = label;
  }
}

class TrackLTF final
{
 public:
  TrackLTF() = default;
  ~TrackLTF() = default;
  void setPoint(const Float_t x, const Float_t y, const Float_t z, const Int_t layer, const Int_t clusterId, const MCCompLabel label, Bool_t& newPoint);
  const Int_t getNPoints() const { return mNPoints; }
  const std::array<Float_t, constants::mft::LayersNumber>& getXCoordinates() const { return mX; }
  const std::array<Float_t, constants::mft::LayersNumber>& getYCoordinates() const { return mY; }
  const std::array<Float_t, constants::mft::LayersNumber>& getZCoordinates() const { return mZ; }
  const std::array<Int_t, constants::mft::LayersNumber>& getLayers() const { return mLayer; }
  const std::array<Int_t, constants::mft::LayersNumber>& getClustersId() const { return mClusterId; }
  const std::array<MCCompLabel, constants::mft::LayersNumber>& getMCCompLabels() const { return mMCCompLabels; }

 private:
  Int_t mNPoints = 0;
  std::array<Float_t, constants::mft::LayersNumber> mX = {-15., -15., -15., -15., -15., -15., -15., -15., -15., -15.};
  std::array<Float_t, constants::mft::LayersNumber> mY = {-15., -15., -15., -15., -15., -15., -15., -15., -15., -15.};
  std::array<Float_t, constants::mft::LayersNumber> mZ = {-80., -80., -80., -80., -80., -80., -80., -80., -80., -80.};
  std::array<Int_t, constants::mft::LayersNumber> mLayer;
  std::array<Int_t, constants::mft::LayersNumber> mClusterId;
  std::array<MCCompLabel, constants::mft::LayersNumber> mMCCompLabels;
  ClassDefNV(TrackLTF, 1);
};

inline void TrackLTF::setPoint(const Float_t x, const Float_t y, const Float_t z, const Int_t layer, const Int_t clusterId, const MCCompLabel label, Bool_t& newPoint)
{
  if (newPoint) {
    if (mNPoints == constants::mft::LayersNumber) {
      LOG(WARN) << "MFT TrackLTF Overflow";
      return;
    }
    mX[mNPoints] = x;
    mY[mNPoints] = y;
    mZ[mNPoints] = z;
    mLayer[mNPoints] = layer;
    mClusterId[mNPoints] = clusterId;
    mMCCompLabels[mNPoints] = label;
    mNPoints++;
  } else {
    mX[mNPoints] = x;
    mY[mNPoints] = y;
    mZ[mNPoints] = z;
    mLayer[mNPoints] = layer;
    mClusterId[mNPoints] = clusterId;
    mMCCompLabels[mNPoints] = label;
  }
}

} // namespace mft
} // namespace o2
#endif /* O2_MFT_TRACKCA_H_ */
