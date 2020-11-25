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

namespace o2
{
namespace mft
{

class Road final
{
 public:
  Road();

  void setPoint(const Int_t layer, const Int_t clusterId)
  {
    mClusterId[layer].push_back(clusterId);
  }

  void setRoadId(const Int_t id) { mRoadId = id; }
  const Int_t getRoadId() const { return mRoadId; }

  const Int_t getNPointsInLayer(Int_t layer) const;

  void getLength(Int_t& layer1, Int_t& layer2) const;

  const std::vector<Int_t>& getClustersIdInLayer(Int_t layer) const { return mClusterId[layer]; }

  template <typename... T>
  Cell& addCellInLayer(Int_t layer, T&&... args);

  std::vector<Cell>& getCellsInLayer(Int_t);

  void addLeftNeighbourToCell(const Int_t, const Int_t, const Int_t, const Int_t);
  void addRightNeighbourToCell(const Int_t, const Int_t, const Int_t, const Int_t);

  void incrementCellLevel(const Int_t, const Int_t);
  void updateCellLevel(const Int_t, const Int_t);

  void setCellLevel(const Int_t, const Int_t, const Int_t);
  const Int_t getCellLevel(const Int_t, const Int_t) const;
  void setCellUsed(const Int_t, const Int_t, const Bool_t);
  const Bool_t isCellUsed(const Int_t, const Int_t) const;

 private:
  Int_t mRoadId;
  std::array<std::vector<Int_t>, constants::mft::LayersNumber> mClusterId;
  std::array<std::vector<Cell>, (constants::mft::LayersNumber - 1)> mCell;
};

inline Road::Road()
  : mRoadId{0}
{
  // Nothing to do
}

inline const Int_t Road::getNPointsInLayer(Int_t layer) const
{
  return mClusterId[layer].size();
}

inline void Road::getLength(Int_t& layer1, Int_t& layer2) const
{
  layer1 = -1, layer2 = 10;
  for (Int_t layer = 0; layer < constants::mft::LayersNumber; ++layer) {
    if (mClusterId[layer].size() > 0) {
      if (layer1 < 0) {
        layer1 = layer;
      }
      layer2 = layer;
    }
  }
}

template <typename... T>
Cell& Road::addCellInLayer(Int_t layer, T&&... values)
{
  mCell[layer].emplace_back(layer, std::forward<T>(values)...);
  return mCell[layer].back();
}

inline std::vector<Cell>& Road::getCellsInLayer(Int_t layer)
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

} // namespace mft
} // namespace o2

#endif /* O2_MFT_ROAD_H_ */
