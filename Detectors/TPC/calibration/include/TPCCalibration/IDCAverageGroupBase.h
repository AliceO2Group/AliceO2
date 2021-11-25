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

/// \file IDCAverageGroupBase.h
/// \brief base class for averaging and grouping of IDCs
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_IDCAVERAGEGROUPBASE_H_
#define ALICEO2_IDCAVERAGEGROUPBASE_H_

#include <vector>
#include "TPCCalibration/IDCGroup.h"
#include "TPCCalibration/RobustAverage.h"
#include "TPCCalibration/IDCGroupHelperSector.h"
#include "TPCCalibration/IDCContainer.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/Sector.h"

namespace o2::tpc
{

/// Helper class for either perform the grouping or draw the grouping
template <class Type>
class IDCAverageGroupBase;

/// dummy class for specializing the class
class IDCAverageGroupCRU;
class IDCAverageGroupTPC;
class IDCAverageGroupDraw;

/// class for averaging and grouping only one CRU
template <>
class IDCAverageGroupBase<IDCAverageGroupCRU>
{
 public:
  /// constructor
  /// \param groupPads number of pads in pad direction which will be grouped
  /// \param groupRows number of pads in row direction which will be grouped
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  /// \param region region of the TPC
  /// \param sector processed sector
  /// \param nThreads number of CPU threads used
  IDCAverageGroupBase(const unsigned char groupPads, const unsigned char groupRows, const unsigned char groupLastRowsThreshold, const unsigned char groupLastPadsThreshold, const unsigned int region, const Sector sector, const int nThreads)
    : mIDCsGrouped{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, region}, mSector{sector}, mRobustAverage(nThreads){};

  /// \return returns number of integration intervals for stored ungrouped IDCs
  unsigned int getNIntegrationIntervals() const { return mIDCsUngrouped.size() / Mapper::PADSPERREGION[getRegion()]; }

  /// \return returns the region of the IDCs
  unsigned int getRegion() const { return mIDCsGrouped.getRegion(); }

  /// \return returns the CRU number
  unsigned int getCRU() const { return getRegion() + mSector * Mapper::NREGIONS; }

  /// \return returns sector of which the IDCs are averaged and grouped
  Sector getSector() const { return mSector; }

  /// setting the ungrouped IDCs using copy constructor
  /// \param idcs vector containing the ungrouped IDCs
  void setIDCs(const std::vector<float>& idcs);

  /// setting the ungrouped IDCs using move semantics
  /// \param idcs vector containing the ungrouped IDCs
  void setIDCs(std::vector<float>&& idcs);

  /// \return returns grouped IDC object using move semantics
  auto getIDCGroupData() && { return std::move(mIDCsGrouped).getData(); }

  /// \return returns the stored ungrouped IDC value for global ungrouped pad row and ungrouped pad
  /// \param ugrow ungrouped global row
  /// \param upad ungrouped pad in pad direction
  /// \param integrationInterval integration interval for which the IDCs will be returned
  float getUngroupedIDCValGlobal(const unsigned int ugrow, const unsigned int upad, const unsigned int integrationInterval) const { return mIDCsUngrouped[getUngroupedIndexGlobal(ugrow, upad, integrationInterval)]; }

  /// \return returns index to data from ungrouped pad and row
  /// \param ulrow ungrouped local row in region
  /// \param upad ungrouped pad in pad direction
  unsigned int getUngroupedIndex(const unsigned int ulrow, const unsigned int upad, const unsigned int integrationInterval) const { return integrationInterval * Mapper::PADSPERREGION[getRegion()] + Mapper::OFFSETCRULOCAL[getRegion()][ulrow] + upad; }

  /// \return returns index to data from ungrouped pad and row
  /// \param ugrow ungrouped global row
  /// \param upad ungrouped pad in pad direction
  unsigned int getUngroupedIndexGlobal(const unsigned int ugrow, const unsigned int upad, const unsigned int integrationInterval) const { return integrationInterval * Mapper::PADSPERREGION[getRegion()] + Mapper::OFFSETCRUGLOBAL[ugrow] + upad; }

  /// draw ungrouped IDCs
  /// \param integrationInterval integration interval for which the IDCs will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawUngroupedIDCs(const unsigned int integrationInterval = 0, const std::string filename = "IDCsUngrouped.pdf") const;

  /// draw grouped IDCs
  /// \param integrationInterval integration interval for which the IDCs will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawGroupedIDCs(const unsigned int integrationInterval = 0, const std::string filename = "IDCsGrouped.pdf") const { mIDCsGrouped.draw(integrationInterval, filename); }

 protected:
  IDCGroup mIDCsGrouped{};                   ///< grouped and averaged IDC values
  std::vector<RobustAverage> mRobustAverage; ///<! object for averaging (each thread will get his one object)
  std::vector<float> mWeightsPad{};          ///< storage of the weights in pad direction used if mOverlapPads>0
  std::vector<float> mWeightsRow{};          ///< storage of the weights in row direction used if mOverlapRows>0
  std::vector<float> mIDCsUngrouped{};       ///< integrated ungrouped IDC values per pad
  const Sector mSector{};                    ///< sector of averaged and grouped IDCs (used for debugging)
};

/// class for averaging and grouping the DeltaIDCs (grouping of A- and C-side)
template <>
class IDCAverageGroupBase<IDCAverageGroupTPC>
{
 public:
  /// constructor
  /// \param groupPads number of pads in pad direction which will be grouped
  /// \param groupRows number of pads in row direction which will be grouped
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  /// \param nThreads number of CPU threads used
  IDCAverageGroupBase(const std::array<unsigned char, Mapper::NREGIONS>& groupPads, const std::array<unsigned char, Mapper::NREGIONS>& groupRows, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold, const int nThreads)
    : mIDCGroupHelperSector(groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold), mRobustAverage(nThreads){};

  /// \return returns number of integration intervalls stored in this object
  /// \param side side of the TPC
  unsigned int getNIntegrationIntervals(const o2::tpc::Side side) const { return mIDCsUngrouped.getIDCDelta(side).size() / Mapper::getNumberOfPadsPerSide(); }

  /// \return returns grouped IDCDelta object
  const auto& getIDCGroupData() & { return std::move(mIDCsGrouped); }

  /// \return returns grouped IDCDelta object
  auto getIDCGroupData() && { return std::move(mIDCsGrouped); }

  /// \return returns helper object containing the grouping parameters and accessing of data indices
  auto& getIDCGroupHelperSector() const { return mIDCGroupHelperSector; }

  /// \return returns index to data from ungrouped pad and row
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  unsigned int getUngroupedIndexGlobal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const { return IDCGroupHelperSector::getUngroupedIndexGlobal(sector, region, urow, upad, integrationInterval); }

  /// \return returns the stored DeltaIDC value for local ungrouped pad row and ungrouped pad
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  float getGroupedIDCDeltaVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const { return mIDCsGrouped.getValue(Sector(sector).side(), mIDCGroupHelperSector.getIndexUngrouped(sector, region, urow, upad, integrationInterval)); }

  /// \return returns the stored DeltaIDC value for local ungrouped pad row and ungrouped pad
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  float getUngroupedIDCDeltaVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const { return mIDCsUngrouped.getValue(Sector(sector).side(), getUngroupedIndexGlobal(sector, region, urow, upad, integrationInterval)); }

  /// draw IDCDelta for one sector for one integration interval
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawGroupedIDCsSector(const unsigned int sector, const unsigned int integrationInterval, const std::string filename = "IDCDeltaGroupedSector.pdf") const { drawIDCDeltaHelper(false, Sector(sector), integrationInterval, true, filename); }

  /// draw IDCDelta for one sector for one integration interval
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawUngroupedIDCsSector(const unsigned int sector, const unsigned int integrationInterval, const std::string filename = "IDCDeltaUngroupedSector.pdf") const { drawIDCDeltaHelper(false, Sector(sector), integrationInterval, false, filename); }

  /// draw IDCs for one side for one integration interval
  /// \param side side which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawGroupedIDCsSide(const o2::tpc::Side side, const unsigned int integrationInterval, const std::string filename = "IDCDeltaGroupedSide.pdf") const { drawIDCDeltaHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), integrationInterval, true, filename); }

  /// draw IDCs for one side for one integration interval
  /// \param side side which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawUngroupedIDCsSide(const o2::tpc::Side side, const unsigned int integrationInterval, const std::string filename = "IDCDeltaUngroupedSide.pdf") const { drawIDCDeltaHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), integrationInterval, false, filename); }

  /// setting the ungrouped IDCs using copy constructor
  /// \param idcs vector containing the ungrouped IDCs
  void setIDCs(const IDCDelta<float>& idcs);

  /// setting the ungrouped IDCs using move semantics
  /// \param idcs vector containing the ungrouped IDCs
  void setIDCs(IDCDelta<float>&& idcs);

 protected:
  IDCDelta<float> mIDCsGrouped{};                                 ///< grouped and averaged IDC values
  IDCGroupHelperSector mIDCGroupHelperSector;                     ///< helper object containing the grouping parameter and methods for accessing data indices
  std::vector<RobustAverage> mRobustAverage;                      ///<! object for averaging (each thread will get his one object)
  std::array<std::vector<float>, Mapper::NREGIONS> mWeightsPad{}; ///< storage of the weights in pad direction used if mOverlapPads>0
  std::array<std::vector<float>, Mapper::NREGIONS> mWeightsRow{}; ///< storage of the weights in row direction used if mOverlapRows>0
  IDCDelta<float> mIDCsUngrouped{};                               ///< integrated ungrouped IDC values per pad

  /// set correct size for grouped IDCs
  void resizeGroupedIDCs();

  /// set correct size for grouped IDCs
  /// \param side side of the TPC
  void resizeGroupedIDCs(const Side side);

  /// helper function for drawing IDCDelta
  void drawIDCDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const bool grouped, const std::string filename) const;
};

} // namespace o2::tpc

#endif
