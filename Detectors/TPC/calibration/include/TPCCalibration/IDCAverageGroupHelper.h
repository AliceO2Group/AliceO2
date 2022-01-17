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

/// \file IDCAverageGroupHelper.h
/// \brief helper class for averaging and grouping of IDCs
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_IDCAVERAGEGROUPHELPER_H_
#define ALICEO2_IDCAVERAGEGROUPHELPER_H_

#include <vector>
#include "TPCBase/Mapper.h"
#include "TPCCalibration/IDCGroupHelperRegion.h"
#include "TPCCalibration/IDCGroupHelperSector.h"
#include "TPCCalibration/IDCGroup.h"

// forward declaration
class TH2Poly;

namespace o2::tpc
{

// forward declaration of some classes
class RobustAverage;
class IDCAverageGroupDraw;
class IDCAverageGroupCRU;
class IDCAverageGroupTPC;
class PadRegionInfo;

template <typename DataT>
struct IDCDelta;

template <class Type>
class IDCAverageGroupHelper;

/// Helper class for performing the grouping per CRU
template <>
class IDCAverageGroupHelper<IDCAverageGroupCRU>
{
 public:
  /// constructor
  IDCAverageGroupHelper(IDCGroup& idcsGrouped, const std::vector<float>& weightsPad, const std::vector<float>& weightsRow, const std::vector<float>& idcsUngrouped, std::vector<RobustAverage>& robustAverage, const unsigned int cru) : mIDCsGrouped{idcsGrouped}, mWeightsPad{weightsPad}, mWeightsRow{weightsRow}, mIDCsUngrouped{idcsUngrouped}, mRobustAverage{robustAverage}, mCRU{cru} {};

  /// \return returns processed region
  unsigned int getRegion() const { return mIDCsGrouped.getRegion(); }

  /// \return returns processed CRU
  auto getCRU() const { return mCRU; }

  /// \return returns grouping parameter
  int getGroupRows() const { return static_cast<int>(mIDCsGrouped.getGroupRows()); }

  /// \return returns grouping parameter
  int getGroupPads() const { return static_cast<int>(mIDCsGrouped.getGroupPads()); }

  /// \return returns offset of the index for a pad which is not grouped (in the region where mGroupPadsSectorEdges is true).
  int getOffsetForEdgePad(const unsigned int upad, const unsigned int ulrow) const { return mIDCsGrouped.getOffsetForEdgePad(upad, ulrow); }

  /// \return returns the number of pads at the sector edges which are not grouped
  unsigned int getTotalGroupPadsSectorEdges() const { return mIDCsGrouped.getTotalGroupPadsSectorEdges(); }

  /// \return returns last ungrouped row
  int getLastRow() const { return static_cast<int>(mIDCsGrouped.getLastRow()); }

  /// \return returns number of grouped pads per row
  /// \param glrow grouped local row
  unsigned int getPadsPerRow(const unsigned int glrow) const { return mIDCsGrouped.getPadsPerRow(glrow); }

  /// \return returns last ungrouped pad for given global row
  /// \param ulrow ungrouped local row
  unsigned int getLastPad(const unsigned int ulrow) const { return mIDCsGrouped.getLastPad(ulrow); }

  /// \return returns weighting in pad direction for nth outer pad
  /// \param relPosPad distance in pads to the group of IDCs
  float getWeightPad(const unsigned int relPosPad) const { return mWeightsPad[relPosPad]; }

  /// \return returns weighting in row direction for nth outer pad
  /// \param relPosRow distance in pads to the group of IDCs
  float getWeightRow(const unsigned int relPosRow) const { return mWeightsRow[relPosRow]; }

  /// \return returns weighting in pad or row direction
  /// \param relPosRow distance in pads to the group of IDCs in row direction
  /// \param relPosPad distance in pads to the group of IDCs in pad direction
  float getWeight(const unsigned int relPosRow, const unsigned int relPosPad) const { return (relPosRow > relPosPad) ? getWeightRow(relPosRow) : getWeightPad(relPosPad); }

  /// \return returns ungrouped IDC value
  /// \param padInRegion local pad number in processed region
  float getUngroupedIDCVal(const unsigned int padInRegion) const { return mIDCsUngrouped[mOffsetUngrouped + padInRegion]; }

  /// \return returns the stored grouped IDC value for local ungrouped pad row and ungrouped pad
  /// \param ugrow global ungrouped row
  /// \param upad pad number of the ungrouped IDCs
  float getGroupedIDCValGlobal(unsigned int ugrow, unsigned int upad) const { return mIDCsGrouped.getValUngroupedGlobal(ugrow, upad, mIntegrationInterval); }

  /// add a value to the averaging object
  /// \param padInRegion pad index in the processed region to the value which will be added
  /// \param weight weight of the value
  void addValue(const unsigned int padInRegion, const float weight);

  /// calculating and setting the grouped IDC value
  /// \param rowGrouped grouped row index
  /// \param padGrouped grouped pad index
  void setGroupedIDC(const unsigned int rowGrouped, const unsigned int padGrouped) { mIDCsGrouped(rowGrouped, padGrouped, mIntegrationInterval) = getGroupedIDC(); }

  /// \return returns grouped and averaged IDC value
  float getGroupedIDC() const;

  /// setting the members for correct data access
  /// \param threadNum thread index
  /// \param integrationInterval integration interval
  void set(const unsigned int threadNum, const unsigned int integrationInterval);

  /// \return returns processed integration interval
  unsigned int getIntegrationInterval() const { return mIntegrationInterval; }

  /// clearing the object for averaging
  void clearRobustAverage();

  /// setting the IDC value at the edge of the sector
  /// \param ulrow ungrouped local row
  /// \param upad ungrouped pad
  /// \param val value which will be stored
  void setSectorEdgeIDC(const unsigned int ulrow, const unsigned int upad) { mIDCsGrouped.setValUngrouped(ulrow, upad, mIntegrationInterval, getGroupedIDC()); }

  /// \return returns which type of grouping is performed for the sector edge pads
  EdgePadGroupingMethod getEdgePadGroupingType() const { return mIDCsGrouped.getEdgePadGroupingType(); }

  /// \return returns the number of differently grouped pads per row: returns sum of digits in integer whoch are not 0 (example: 0: returns 0, 11: returns 2, 321: returns 3)
  unsigned int getGroupedPadsSectorEdges() const { return mIDCsGrouped.getGroupedPadsSectorEdges(); }

  /// \return returns number of ungrouped pads for grouped pad (example groupPadsSectorEdges=324530: group=0 -> 3, group=1 -> 5, group=2 -> 4...
  /// \param group index of the group
  unsigned int getPadsInGroupSectorEdges(const unsigned indexGroup) const { return mIDCsGrouped.getPadsInGroupSectorEdges(indexGroup); }

 private:
  IDCGroup& mIDCsGrouped;                     ///< grouped and averaged IDC values
  const std::vector<float>& mWeightsPad{};    ///< storage of the weights in pad direction used if mOverlapPads>0
  const std::vector<float>& mWeightsRow{};    ///< storage of the weights in row direction used if mOverlapRows>0
  const std::vector<float>& mIDCsUngrouped{}; ///< integrated ungrouped IDC values per pad
  std::vector<RobustAverage>& mRobustAverage; ///<! object for averaging (each thread will get his one object)
  const unsigned int mCRU{};                  ///< cru of the processed region
  unsigned int mThreadNum{};                  ///< thread number for robust averaging
  unsigned int mIntegrationInterval{};        ///< current integration interval
  unsigned int mOffsetUngrouped{};            ///< offset to calculate the index for the ungrouped IDCs
};

template <>
class IDCAverageGroupHelper<IDCAverageGroupTPC>
{
 public:
  IDCAverageGroupHelper(IDCDelta<float>& idcsGrouped, const std::array<std::vector<float>, Mapper::NREGIONS>& weightsPad, const std::array<std::vector<float>, Mapper::NREGIONS>& weightsRow, const IDCDelta<float>& idcsUngrouped, std::vector<RobustAverage>& robustAverage, const IDCGroupHelperSector& idcGroupHelperSector) : mIDCsGrouped{idcsGrouped}, mWeightsPad{weightsPad}, mWeightsRow{weightsRow}, mIDCsUngrouped{idcsUngrouped}, mRobustAverage{robustAverage}, mIDCGroupHelperSector{idcGroupHelperSector} {};

  /// \return returns processed region
  unsigned int getRegion() const { return mCRU.region(); }

  /// \return returns processed CRU
  auto getCRU() const { return mCRU; }

  /// \return returns processed sector
  Sector getSector() const { return mCRU.sector(); }

  /// \return returns processed side of the TPC
  Side getSide() const { return mCRU.side(); }

  /// \return returns grouping parameter
  int getGroupRows() const { return static_cast<int>(mIDCGroupHelperSector.getGroupingParameter().getGroupRows(getRegion())); }

  /// \return returns grouping parameter
  int getGroupPads() const { return static_cast<int>(mIDCGroupHelperSector.getGroupingParameter().getGroupPads(getRegion())); }

  /// \return returns offset of the index for a pad which is not grouped (in the region where mGroupPadsSectorEdges is true).
  int getOffsetForEdgePad(const unsigned int upad, const unsigned int ulrow) const { return mIDCGroupHelperSector.getOffsetForEdgePad(upad, ulrow, getRegion()); }

  /// \return returns the number of pads at the sector edges which are not grouped
  unsigned int getTotalGroupPadsSectorEdges() const { return mIDCGroupHelperSector.getGroupingParameter().getTotalGroupPadsSectorEdges(); }

  /// \return returns last ungrouped row
  int getLastRow() const { return static_cast<int>(mIDCGroupHelperSector.getLastRow(getRegion())); }

  /// \return returns number of grouped pads per row
  /// \param glrow grouped local row
  unsigned int getPadsPerRow(const unsigned int glrow) const { return mIDCGroupHelperSector.getPadsPerRow(getRegion(), glrow); }

  /// \return returns last ungrouped pad for given global row
  /// \param ulrow ungrouped local row
  unsigned int getLastPad(const unsigned int ulrow) const { return mIDCGroupHelperSector.getLastPad(getRegion(), ulrow); }

  /// \return returns weighting in pad direction for nth outer pad
  /// \param relPosPad distance in pads to the group of IDCs
  float getWeightPad(const unsigned int relPosPad) const { return mWeightsPad[getRegion()][relPosPad]; }

  /// \return returns weighting in row direction for nth outer pad
  /// \param relPosRow distance in pads to the group of IDCs
  float getWeightRow(const unsigned int relPosRow) const { return mWeightsRow[getRegion()][relPosRow]; }

  /// \return returns weighting in pad or row direction
  /// \param relPosRow distance in pads to the group of IDCs in row direction
  /// \param relPosPad distance in pads to the group of IDCs in pad direction
  float getWeight(const unsigned int relPosRow, const unsigned int relPosPad) const { return (relPosRow > relPosPad) ? getWeightRow(relPosRow) : getWeightPad(relPosPad); }

  /// \return returns the stored DeltaIDC value for local ungrouped pad row and ungrouped pad
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  float getGroupedIDCValGlobal(unsigned int urow, unsigned int upad) const;

  /// \param padInRegion local pad number in processed region
  float getUngroupedIDCVal(const unsigned int padInRegion) const;

  /// \return returns processed integration interval
  unsigned int getIntegrationInterval() const { return mIntegrationInterval; }

  /// add a value to the averaging object
  /// \param padInRegion pad index in the processed region to the value which will be added
  /// \param weight weight of the value
  void addValue(const unsigned int padInRegion, const float weight);

  /// calculating and setting the grouped IDC value
  /// \param rowGrouped grouped row index
  /// \param padGrouped grouped pad index
  void setGroupedIDC(const unsigned int rowGrouped, const unsigned int padGrouped) { setGroupedIDC(rowGrouped, padGrouped, getGroupedIDC()); }

  /// \return returns grouped and averaged IDC value
  float getGroupedIDC() const;

  /// \param threadNum thread index
  void setThreadNum(const unsigned int threadNum) { mThreadNum = threadNum; }

  /// set integration interval for current processed region
  void setIntegrationInterval(const unsigned int integrationInterval);

  /// set current processed CRU
  void setCRU(const CRU cru) { mCRU = cru; }

  /// clearing the object for averaging
  void clearRobustAverage();

  /// setting the IDC value at the edge of the sector
  /// \param ulrow ungrouped local row
  /// \param upad ungrouped pad
  /// \param val value which will be stored
  void setSectorEdgeIDC(const unsigned int ulrow, const unsigned int upad);

  /// \return returns which type of grouping is performed for the sector edge pads
  EdgePadGroupingMethod getEdgePadGroupingType() const { return mIDCGroupHelperSector.getGroupingParameter().getEdgePadGroupingType(); }

  unsigned int getGroupedPadsSectorEdges() const { return mIDCGroupHelperSector.getGroupingParameter().getGroupedPadsSectorEdges(); }

  /// \return returns number of ungrouped pads for grouped pad (example groupPadsSectorEdges=324530: group=0 -> 3, group=1 -> 5, group=2 -> 4...
  /// \param group index of the group
  unsigned int getPadsInGroupSectorEdges(const unsigned indexGroup) const { return mIDCGroupHelperSector.getGroupingParameter().getPadsInGroupSectorEdges(indexGroup); }

 private:
  IDCDelta<float>& mIDCsGrouped;                                         ///< grouped and averaged IDC values
  const std::array<std::vector<float>, Mapper::NREGIONS>& mWeightsPad{}; ///< storage of the weights in pad direction used if mOverlapPads>0
  const std::array<std::vector<float>, Mapper::NREGIONS>& mWeightsRow{}; ///< storage of the weights in row direction used if mOverlapRows>0
  const IDCDelta<float>& mIDCsUngrouped;                                 ///< integrated ungrouped IDC values per pad
  std::vector<RobustAverage>& mRobustAverage;                            ///<! object for averaging (each thread will get his one object)
  const IDCGroupHelperSector& mIDCGroupHelperSector;                     ///< helper for acces the data
  CRU mCRU{};                                                            ///< cru of the processed region
  unsigned int mThreadNum{};                                             ///< thread number for robust averaging
  unsigned int mIntegrationInterval{};                                   ///< current integration interval
  unsigned int mOffsetUngrouped{};                                       ///< offset to calculate the index for the ungrouped IDCs
  unsigned int mOffsetGrouped{};                                         ///< offset to calculate the index for the grouped IDCs

  /// calculating and setting the grouped IDC value
  /// \param glrow local row of the grouped IDCs
  /// \param pad pad of the grouped IDCs
  /// \param integrationInterval integration interval
  void setGroupedIDC(const unsigned int glrow, const unsigned int padGrouped, const float val);
};

/// Helper class for drawing the IDCs
template <>
class IDCAverageGroupHelper<IDCAverageGroupDraw> : public IDCGroupHelperRegion
{
 public:
  IDCAverageGroupHelper(const unsigned char groupPads, const unsigned char groupRows, const unsigned char groupLastRowsThreshold, const unsigned char groupLastPadsThreshold, const unsigned int groupNotnPadsSectorEdges, const unsigned int region, const unsigned int nPads, const PadRegionInfo& padInf, TH2Poly& poly)
    : IDCGroupHelperRegion{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupNotnPadsSectorEdges, region}, mCountDraw(nPads), mPadInf{padInf}, mPoly{poly} {};

  std::vector<int> mCountDraw;                  ///< counter to keep track of the already drawn pads
  const PadRegionInfo& mPadInf;                 ///< object for storing pad region information
  TH2Poly& mPoly;                               ///< TH2Poly which will be used/filled for drawing
  int mGroupCounter = 0;                        ///< counter for drawing the group index
  int mCol = 0;                                 ///< counter for drawing the color of each group
  const std::array<int, 4> mColors{1, 2, 3, 4}; ///< colors (TH2Poly will be filled with these values)
};

} // namespace o2::tpc

#endif
