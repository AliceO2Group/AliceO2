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
/// \file Road.h
/// \brief A cylindrical volume, around a seed line connecting two outer clusters, inside which the cellular automaton algorithm is applied
///

#ifndef O2_MFT_ROAD_H_
#define O2_MFT_ROAD_H_

#include "MFTTracking/Cell.h"
#include "MFTTracking/Constants.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace MFT
{

class Road final
{
 public:
  Road();
  void setPoint(const Float_t x, const Float_t y, const Float_t z, const Int_t layer, const Int_t clusterId, const MCCompLabel label, Bool_t& newPoint);
  void setRoadId(const Int_t id) { mRoadId = id; }
  const Int_t getRoadId() const { return mRoadId; }
  void setNDisks(const Int_t nd) { mNDisks = nd; }
  const Int_t getNDisks() const { return mNDisks; }
  const Int_t getNPoints() const { return mNPoints; }
  const Int_t getNPointsInLayer(Int_t layer) const;
  void getLength(Int_t& layer1, Int_t& layer2) const;
  void setHasTracksCA() { mHasTracksCA = kTRUE; }
  const Bool_t hasTracksCA() const;

  const std::vector<Float_t>& getXCoordinatesInLayer(Int_t layer) const { return mX[layer]; }
  const std::vector<Float_t>& getYCoordinatesInLayer(Int_t layer) const { return mY[layer]; }
  const std::vector<Float_t>& getZCoordinatesInLayer(Int_t layer) const { return mZ[layer]; }
  const std::vector<Int_t>& getClustersIdInLayer(Int_t layer) const { return mClusterId[layer]; }
  const std::vector<MCCompLabel>& getMCCompLabelsInLayer(Int_t layer) const { return mMCCompLabel[layer]; }

  template <typename... T>
  void addCellToLayer(Int_t layer, T&&... args);

  const std::vector<Cell>& getCellsInLayer(Int_t) const;
  void addLeftNeighbourToCell(const Int_t, const Int_t, const Int_t, const Int_t);
  void addRightNeighbourToCell(const Int_t, const Int_t, const Int_t, const Int_t);
  void incrementCellLevel(const Int_t, const Int_t);
  void updateCellLevel(const Int_t, const Int_t);
  const Int_t getCellLevel(const Int_t, const Int_t) const;
  const Bool_t isCellUsed(const Int_t, const Int_t) const;
  void setCellUsed(const Int_t, const Int_t, const Bool_t);
  void setCellLevel(const Int_t, const Int_t, const Int_t);

 private:
  Int_t mRoadId;
  Int_t mNDisks;
  Int_t mNPoints;
  Bool_t mHasTracksCA;
  std::array<std::vector<Float_t>, Constants::MFT::LayersNumber> mX;
  std::array<std::vector<Float_t>, Constants::MFT::LayersNumber> mY;
  std::array<std::vector<Float_t>, Constants::MFT::LayersNumber> mZ;
  std::array<std::vector<Int_t>, Constants::MFT::LayersNumber> mClusterId;
  std::array<std::vector<MCCompLabel>, Constants::MFT::LayersNumber> mMCCompLabel;
  std::array<std::vector<Cell>, (Constants::MFT::LayersNumber - 1)> mCell;
};

inline Road::Road()
  : mRoadId{ 0 },
    mNDisks{ 0 },
    mNPoints{ 0 }
{
  // Nothing to do
}

inline void Road::setPoint(const Float_t x, const Float_t y, const Float_t z, const Int_t layer, const Int_t clusterId, const MCCompLabel label, Bool_t& newPoint)
{
  if (!newPoint) {
    if (!mZ.empty()) {
      mX[layer].pop_back();
      mY[layer].pop_back();
      mZ[layer].pop_back();
      mClusterId[layer].pop_back();
      mMCCompLabel[layer].pop_back();
    }
  } else { // end replace point
    newPoint = kFALSE;
  }
  mX[layer].push_back(x);
  mY[layer].push_back(y);
  mZ[layer].push_back(z);
  mClusterId[layer].push_back(clusterId);
  mMCCompLabel[layer].emplace_back(label);
  ++mNPoints;
}

inline const Int_t Road::getNPointsInLayer(Int_t layer) const
{
  return mX[layer].size();
}

inline void Road::getLength(Int_t& layer1, Int_t& layer2) const
{
  layer1 = -1, layer2 = 10;
  for (Int_t layer = 0; layer < Constants::MFT::LayersNumber; ++layer) {
    if (mX[layer].size() > 0) {
      if (layer1 < 0) {
        layer1 = layer;
      }
      layer2 = layer;
    }
  }
}

inline const Bool_t Road::hasTracksCA() const
{
  return mHasTracksCA;
}

template <typename... T>
void Road::addCellToLayer(Int_t layer, T&&... values)
{
  mCell[layer].emplace_back(layer, std::forward<T>(values)...);
}

inline const std::vector<Cell>& Road::getCellsInLayer(Int_t layer) const
{
  return mCell[layer];
}

inline void Road::addLeftNeighbourToCell(const Int_t layer, const Int_t cellId, const Int_t layerL, const Int_t cellIdL)
{
  mCell[layer][cellId].addLeftNeighbour(layerL, cellIdL);
}

inline void Road::addRightNeighbourToCell(const Int_t layer, const Int_t cellId, const Int_t layerR, const Int_t cellIdR)
{
  mCell[layer][cellId].addRightNeighbour(layerR, cellIdR);
}

inline void Road::incrementCellLevel(const Int_t layer, const Int_t cellId)
{
  mCell[layer][cellId].incrementLevel();
}

inline void Road::updateCellLevel(const Int_t layer, const Int_t cellId)
{
  mCell[layer][cellId].updateLevel();
}

inline const Int_t Road::getCellLevel(const Int_t layer, const Int_t cellId) const
{
  return mCell[layer][cellId].getLevel();
}

inline const Bool_t Road::isCellUsed(const Int_t layer, const Int_t cellId) const
{
  return mCell[layer][cellId].isUsed();
}

inline void Road::setCellUsed(const Int_t layer, const Int_t cellId, const Bool_t suc)
{
  mCell[layer][cellId].setUsed(suc);
}

inline void Road::setCellLevel(const Int_t layer, const Int_t cellId, const Int_t level)
{
  mCell[layer][cellId].setLevel(level);
}

} // namespace MFT
} // namespace o2

#endif /* O2_MFT_ROAD_H_ */
