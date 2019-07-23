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

#include "SimulationDataFormat/MCCompLabel.h"

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
  const Int_t getNPoints() const { return mX.size(); }
  void setChiSquareZX(const Float_t chisq) { mChiSquareZX = chisq; }
  void setChiSquareZY(const Float_t chisq) { mChiSquareZY = chisq; }
  const Float_t getChiSquareZX() const { return mChiSquareZX; }
  const Float_t getChiSquareZY() const { return mChiSquareZY; }
  const std::vector<Float_t>& getXCoordinates() const { return mX; }
  const std::vector<Float_t>& getYCoordinates() const { return mY; }
  const std::vector<Float_t>& getZCoordinates() const { return mZ; }
  const std::vector<Int_t>& getLayers() const { return mLayer; }
  const std::vector<Int_t>& getClustersId() const { return mClusterId; }
  const std::vector<Int_t>& getCellsLayer() const { return mCellLayer; }
  const std::vector<Int_t>& getCellsId() const { return mCellId; }
  const std::vector<MCCompLabel>& getMCCompLabels() const { return mMCCompLabels; }

 private:
  Int_t mRoadId{ -1 };
  Float_t mChiSquareZX{ 0. };
  Float_t mChiSquareZY{ 0. };
  std::vector<Float_t> mX;
  std::vector<Float_t> mY;
  std::vector<Float_t> mZ;
  std::vector<Int_t> mLayer;
  std::vector<Int_t> mClusterId;
  std::vector<Int_t> mCellLayer;
  std::vector<Int_t> mCellId;
  std::vector<MCCompLabel> mMCCompLabels;
  ClassDefNV(TrackCA, 1)
};

inline void TrackCA::addCell(const Int_t layer, const Int_t cellId)
{
  mCellLayer.push_back(layer);
  mCellId.push_back(cellId);
}

inline void TrackCA::removeLastCell(Int_t& layer, Int_t& cellId)
{
  layer = mCellLayer.back();
  cellId = mCellId.back();

  if (mX.size() == 2) { // we have only a single cell in the track
    mX.pop_back();
    mY.pop_back();
    mZ.pop_back();
    mLayer.pop_back();
    mClusterId.pop_back();
    mMCCompLabels.pop_back();
  }
  mX.pop_back();
  mY.pop_back();
  mZ.pop_back();
  mLayer.pop_back();
  mClusterId.pop_back();
  mMCCompLabels.pop_back();

  mCellLayer.pop_back();
  mCellId.pop_back();
}

inline const Int_t TrackCA::getNCells() const
{
  return mCellId.size();
}

inline const Int_t TrackCA::getRoadId() const
{
  return mRoadId;
}

inline void TrackCA::setPoint(const Float_t x, const Float_t y, const Float_t z, const Int_t layer, const Int_t clusterId, const MCCompLabel label, Bool_t& newPoint)
{
  if (!newPoint) {
    if (!mX.empty()) {
      mX.pop_back();
      mY.pop_back();
      mZ.pop_back();
      mLayer.pop_back();
      mClusterId.pop_back();
      mMCCompLabels.pop_back();
    }
  } else { // end replace point
    newPoint = kFALSE;
  }
  mX.push_back(x);
  mY.push_back(y);
  mZ.push_back(z);
  mLayer.push_back(layer);
  mClusterId.push_back(clusterId);
  mMCCompLabels.emplace_back(label);
}

class TrackLTF final
{
 public:
  TrackLTF() = default;
  ~TrackLTF() = default;
  void setPoint(const Float_t x, const Float_t y, const Float_t z, const Int_t layer, const Int_t clusterId, const MCCompLabel label, Bool_t& newPoint);
  const Int_t getNPoints() const { return mX.size(); }
  const std::vector<Float_t>& getXCoordinates() const { return mX; }
  const std::vector<Float_t>& getYCoordinates() const { return mY; }
  const std::vector<Float_t>& getZCoordinates() const { return mZ; }
  const std::vector<Int_t>& getLayers() const { return mLayer; }
  const std::vector<Int_t>& getClustersId() const { return mClusterId; }
  const std::vector<MCCompLabel>& getMCCompLabels() const { return mMCCompLabels; }

 private:
  std::vector<Float_t> mX;
  std::vector<Float_t> mY;
  std::vector<Float_t> mZ;
  std::vector<Int_t> mLayer;
  std::vector<Int_t> mClusterId;
  std::vector<MCCompLabel> mMCCompLabels;
  ClassDefNV(TrackLTF, 1)
};

inline void TrackLTF::setPoint(const Float_t x, const Float_t y, const Float_t z, const Int_t layer, const Int_t clusterId, const MCCompLabel label, Bool_t& newPoint)
{
  if (!newPoint) {
    if (!mX.empty()) {
      mX.pop_back();
      mY.pop_back();
      mZ.pop_back();
      mLayer.pop_back();
      mClusterId.pop_back();
      mMCCompLabels.pop_back();
    }
  } else { // end replace point
    newPoint = kFALSE;
  }
  mX.push_back(x);
  mY.push_back(y);
  mZ.push_back(z);
  mLayer.push_back(layer);
  mClusterId.push_back(clusterId);
  mMCCompLabels.emplace_back(label);
}

} // namespace MFT
} // namespace o2
#endif /* O2_MFT_TRACKCA_H_ */
