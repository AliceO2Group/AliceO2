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
/// \file Cell.h
/// \brief A segment connecting two clusters from two planes
///

#ifndef O2_MFT_CELL_H_
#define O2_MFT_CELL_H_

#include <array>
#include <vector>
#include <iostream>

#include "MFTTracking/Constants.h"

namespace o2
{
namespace mft
{

class Cell
{
 public:
  Cell();
  /// layer1, layer2, clsInLayer1, clsInLayer2, cellId; set level = 1
  Cell(const Int_t, const Int_t, const Int_t, const Int_t, const Int_t);

  const Int_t getFirstLayerId() const;
  const Int_t getSecondLayerId() const;
  const Int_t getFirstClusterIndex() const;
  const Int_t getSecondClusterIndex() const;
  const Int_t getLevel() const;
  void setLevel(const Int_t);
  void incrementLevel();
  void updateLevel();
  void addRightNeighbour(const Int_t, const Int_t);
  void addLeftNeighbour(const Int_t, const Int_t);
  void setUsed(const Bool_t suc) { mIsUsed = suc; }
  const Bool_t isUsed() const { return mIsUsed; }
  const Int_t getCellId() const { return mCellId; };
  void setCellId(const Int_t);
  const auto& getLeftNeighbours() const { return mLeftNeighbours; };
  const auto& getRightNeighbours() const { return mRightNeighbours; };
  const UChar_t getNLeftNeighbours() const { return mLeftNeighbours.size(); }
  const UChar_t getNRightNeighbours() const { return mRightNeighbours.size(); }

  void setCoordinates(Float_t* coord)
  {
    mCoord[0] = coord[0]; // X1
    mCoord[1] = coord[1]; // Y1
    mCoord[2] = coord[2]; // Z1
    mCoord[3] = coord[3]; // X2
    mCoord[4] = coord[4]; // Y2
    mCoord[5] = coord[5]; // Z2
  }

  const Float_t getX1() const { return mCoord[0]; }
  const Float_t getY1() const { return mCoord[1]; }
  const Float_t getZ1() const { return mCoord[2]; }
  const Float_t getX2() const { return mCoord[3]; }
  const Float_t getY2() const { return mCoord[4]; }
  const Float_t getZ2() const { return mCoord[5]; }

 private:
  const Int_t mFirstLayerId;
  const Int_t mSecondLayerId;
  const Int_t mFirstClusterIndex;
  const Int_t mSecondClusterIndex;
  Int_t mLevel;
  Bool_t mUpdateLevel;
  Bool_t mIsUsed;
  Int_t mCellId;
  std::vector<std::pair<Int_t, Int_t>> mLeftNeighbours;
  std::vector<std::pair<Int_t, Int_t>> mRightNeighbours;
  Float_t mCoord[6];
};

inline Cell::Cell()
  : mFirstLayerId{-1},
    mSecondLayerId{-1},
    mFirstClusterIndex{-1},
    mSecondClusterIndex{-1},
    mLevel{0},
    mUpdateLevel{kFALSE},
    mIsUsed{kFALSE},
    mCellId{-1}
{
  // Default constructor, for the dictionary
}

inline Cell::Cell(const Int_t firstLayerId, const Int_t secondLayerId, const Int_t firstClusterIndex, const Int_t secondClusterIndex, const Int_t cellIndex)
  : mFirstLayerId{firstLayerId},
    mSecondLayerId{secondLayerId},
    mFirstClusterIndex{firstClusterIndex},
    mSecondClusterIndex{secondClusterIndex},
    mLevel{1},
    mUpdateLevel{kFALSE},
    mIsUsed{kFALSE},
    mCellId{cellIndex}
{
}

inline const Int_t Cell::getFirstLayerId() const { return mFirstLayerId; }

inline const Int_t Cell::getSecondLayerId() const { return mSecondLayerId; }

inline const Int_t Cell::getFirstClusterIndex() const { return mFirstClusterIndex; }

inline const Int_t Cell::getSecondClusterIndex() const { return mSecondClusterIndex; }

inline const Int_t Cell::getLevel() const
{
  return mLevel;
}

inline void Cell::setLevel(const Int_t level) { mLevel = level; }

inline void Cell::addRightNeighbour(const Int_t layer, const Int_t clusterId)
{
  mRightNeighbours.emplace_back(layer, clusterId);
}

inline void Cell::addLeftNeighbour(const Int_t layer, const Int_t clusterId)
{
  mLeftNeighbours.emplace_back(layer, clusterId);
}

inline void Cell::incrementLevel()
{
  mUpdateLevel = kTRUE;
}

inline void Cell::updateLevel()
{
  if (mUpdateLevel) {
    ++mLevel;
    mUpdateLevel = kFALSE;
  }
}

} // namespace mft
} // namespace o2
#endif /* O2_MFT_CELL_H_ */
