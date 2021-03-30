// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file IDCGroupHelperRegion.h
/// \brief helper class for grouping of pads and rows for one region
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_IDCGROUPHELPERREGION_H_
#define ALICEO2_TPC_IDCGROUPHELPERREGION_H_

#include <vector>
#include "Rtypes.h"

namespace o2::tpc
{

/// Helper class for accessing grouped pads for one region

class IDCGroupHelperRegion
{
 public:
  /// constructor
  /// \param groupPads number of pads in pad direction which will be grouped
  /// \param groupRows number of pads in row direction which will be grouped
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  /// \param region region of the TPC
  IDCGroupHelperRegion(const unsigned char groupPads = 4, const unsigned char groupRows = 4, const unsigned char groupLastRowsThreshold = 2, const unsigned char groupLastPadsThreshold = 2, const unsigned int region = 0)
    : mGroupPads{groupPads}, mGroupRows{groupRows}, mGroupLastRowsThreshold{groupLastRowsThreshold}, mGroupLastPadsThreshold{groupLastPadsThreshold}, mRegion{region}
  {
    initIDCGroupHelperRegion();
  }

  /// \return returns number of grouped rows
  unsigned int getNRows() const { return mRows; }

  /// \return returns number of grouped pads
  /// \param row grouped local row
  unsigned int getPadsPerRow(const unsigned int glrow) const { return mPadsPerRow[glrow]; }

  /// \return returns number of grouped pads for all rows
  const std::vector<unsigned int>& getPadsPerRow() const { return mPadsPerRow; }

  /// \return returns offsets for rows to calculate data index
  const std::vector<unsigned int>& getRowOffset() const { return mOffsRow; }

  /// \return returns the number of pads in pad direction which are grouped
  unsigned int getGroupPads() const { return mGroupPads; }

  /// \return returns the number of pads in row direction which are grouped
  unsigned int getGroupRows() const { return mGroupRows; }

  /// \return returns threshold for grouping the last group in row direction
  unsigned int getGroupLastRowsThreshold() const { return mGroupLastRowsThreshold; }

  /// \return returns threshold for grouping the last group in pad direction
  unsigned int getGroupLastPadsThreshold() const { return mGroupLastPadsThreshold; }

  /// \return returns the region for which the IDCs are stored
  unsigned int getRegion() const { return mRegion; }

  /// \returns returns number of IDCS per integration interval
  unsigned int getNIDCsPerIntegrationInterval() const { return mNIDCsPerCRU; }

  /// \return returns the row of the group from the local ungrouped row in a region
  /// \param ulrow local ungrouped row in a region
  /// \param groupRows grouping parameter for number of pads in row direction which are grouped
  /// \param groupedrows number of grouped rows
  static unsigned int getGroupedRow(const unsigned int ulrow, const unsigned int groupRows, const unsigned int groupedrows);

  /// \return returns the row of the group from the local ungrouped row in a region
  /// \param ulrow local ungrouped row in a region
  unsigned int getGroupedRow(const unsigned int ulrow) const { return getGroupedRow(ulrow, mGroupRows, mRows); }

  /// \return returns the grouped pad index from ungrouped pad and row
  /// \param upad ungrouped pad
  /// \param ulrow local ungrouped row in a region
  /// \param region region
  /// \param groupPads grouping parameter for number of pads in pad direction which are grouped
  /// \param groupRows grouping parameter for number of pads in row direction which are grouped
  /// \param groupedrows number of grouped rows
  /// \param padsPerRow vector containing the number of pads per row
  static unsigned int getGroupedPad(const unsigned int upad, const unsigned int ulrow, const unsigned int region, const unsigned int groupPads, const unsigned int groupRows, const unsigned int groupedrows, const std::vector<unsigned int>& padsPerRow);

  /// \return returns the grouped pad index from ungrouped pad and row
  /// \param pad ungrouped pad
  /// \param lrow local ungrouped row in a region
  unsigned int getGroupedPad(const unsigned int pad, const unsigned int ulrow) const { return getGroupedPad(pad, ulrow, mRegion, mGroupPads, mGroupRows, mRows, mPadsPerRow); };

  /// \return returns index to the data
  /// \param row local row of the grouped IDCs
  /// \param pad pad of the grouped IDCs
  /// \param integrationInterval integration interval
  unsigned int getIndex(const unsigned int glrow, const unsigned int pad, unsigned int integrationInterval) const { return mNIDCsPerCRU * integrationInterval + mOffsRow[glrow] + pad; }

  /// \return returns index to the data
  /// \param urow local ungrouped row
  /// \param upad ungrouped pad
  /// \param integrationInterval integration interval
  unsigned int getIndexUngrouped(const unsigned int ulrow, const unsigned int upad, unsigned int integrationInterval) const { return getIndex(getGroupedRow(ulrow), getGroupedPad(upad, ulrow), integrationInterval); }

  /// \return returns the global pad number for given local pad row and pad
  /// \param ulrow local ungrouped row in a region
  /// \param pad ungrouped pad in row
  unsigned int getGlobalPadNumber(const unsigned int ulrow, const unsigned int pad) const;

  /// \return returns last ungrouped row
  unsigned int getLastRow() const;

  /// \return returns last ungrouped pad for given global row
  /// \param row local ungrouped row
  unsigned int getLastPad(const unsigned int ulrow) const;

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "IDCGroupHelperRegion.root", const char* outName = "IDCGroupHelperRegion") const;

 protected:
  const unsigned char mGroupPads{};              ///< grouping parameter in pad direction (how many pads in pad direction are grouped)
  const unsigned char mGroupRows{};              ///< grouping parameter in pad direction (how many pads in pad direction are grouped)
  const unsigned char mGroupLastRowsThreshold{}; ///< if the last group (region edges) consists in row direction less then mGroupLastRowsThreshold pads then it will be grouped into the previous group
  const unsigned char mGroupLastPadsThreshold{}; ///< if the last group (sector edges) consists in pad direction less then mGroupLastPadsThreshold pads then it will be grouped into the previous group
  const unsigned int mRegion{};                  ///< region of input IDCs
  unsigned int mNIDCsPerCRU{};                   ///< total number of IDCs per CRU per integration interval
  unsigned int mRows{};                          ///< number of grouped rows
  std::vector<unsigned int> mPadsPerRow{};       ///< number of grouped pads per grouped row
  std::vector<unsigned int> mOffsRow{};          ///< offset to calculate the index in the data from grouped row and grouped pad

  /// set number of grouped rows
  void setRows(const unsigned int nRows);

  /// initialize members
  void initIDCGroupHelperRegion();

  ClassDefNV(IDCGroupHelperRegion, 1)
};

} // namespace o2::tpc

#endif
