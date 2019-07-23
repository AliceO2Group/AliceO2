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
/// \file Cell.h
/// \brief A segment connecting two clusters from two planes
///

#ifndef O2_MFT_CELL_H_
#define O2_MFT_CELL_H_

#include <array>
#include <vector>

namespace o2
{
namespace mft
{

class Cell final
{
 public:
  Cell();
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
  const std::vector<std::pair<Int_t, Int_t>>& getLeftNeighbours() const;
  const std::vector<std::pair<Int_t, Int_t>>& getRightNeighbours() const;

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
};

inline Cell::Cell()
  : mFirstLayerId{ -1 },
    mSecondLayerId{ -1 },
    mFirstClusterIndex{ -1 },
    mSecondClusterIndex{ -1 },
    mLevel{ 0 },
    mUpdateLevel{ kFALSE },
    mIsUsed{ kFALSE },
    mCellId{ -1 }
{
  // Default constructor, for the dictionary
}

inline Cell::Cell(const Int_t firstLayerId, const Int_t secondLayerId, const Int_t firstClusterIndex, const Int_t secondClusterIndex, const Int_t cellIndex)
  : mFirstLayerId{ firstLayerId },
    mSecondLayerId{ secondLayerId },
    mFirstClusterIndex{ firstClusterIndex },
    mSecondClusterIndex{ secondClusterIndex },
    mLevel{ 1 },
    mUpdateLevel{ kFALSE },
    mIsUsed{ kFALSE },
    mCellId{ cellIndex }
{
  // Nothing to do
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

inline const std::vector<std::pair<Int_t, Int_t>>& Cell::getLeftNeighbours() const
{
  return mLeftNeighbours;
}

inline const std::vector<std::pair<Int_t, Int_t>>& Cell::getRightNeighbours() const
{
  return mRightNeighbours;
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

} // namespace MFT
} // namespace o2
#endif /* O2_MFT_CELL_H_ */
