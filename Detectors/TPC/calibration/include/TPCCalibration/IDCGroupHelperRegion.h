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

/// \file IDCGroupHelperRegion.h
/// \brief helper class for grouping of pads and rows for one region
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_IDCGROUPHELPERREGION_H_
#define ALICEO2_TPC_IDCGROUPHELPERREGION_H_

#include <vector>
#include "Rtypes.h"

namespace o2::tpc
{

enum class EdgePadGroupingMethod : char;

/// Helper class for accessing grouped pads for one region
class IDCGroupHelperRegion
{
 public:
  /// constructor
  /// \param groupPads number of pads in pad direction which will be grouped
  /// \param groupRows number of pads in row direction which will be grouped
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  /// \param groupNotnPadsSectorEdges decoded number of pads at the sector edges which are not grouped (example: 0: no pads are grouped, 11: first two pads are not grouped, 321: first pad is not grouped, second + third pads are grouped, fourth + fifth + sixth pads are grouped)
  /// \param region region of the TPC
  IDCGroupHelperRegion(const unsigned char groupPads, const unsigned char groupRows, const unsigned char groupLastRowsThreshold, const unsigned char groupLastPadsThreshold, const unsigned int groupNotnPadsSectorEdges, const unsigned int region)
    : mGroupPads{groupPads}, mGroupRows{groupRows}, mGroupLastRowsThreshold{groupLastRowsThreshold}, mGroupLastPadsThreshold{groupLastPadsThreshold}, mGroupPadsSectorEdges{groupNotnPadsSectorEdges}, mRegion{region}
  {
    initIDCGroupHelperRegion();
  }

  /// default constructor for ROOT I/O
  IDCGroupHelperRegion() = default;

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

  /// \return returns the number of pads at the sector edges which are not grouped
  unsigned int getGroupPadsSectorEdges() const { return mGroupPadsSectorEdges; }

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
  static unsigned int getGroupedPad(const unsigned int upad, const unsigned int ulrow, const unsigned int region, const unsigned int groupPads, const unsigned int groupRows, const unsigned int groupedrows, const unsigned int groupPadsSectorEdges, const std::vector<unsigned int>& padsPerRow);

  /// \return returns the grouped pad index from ungrouped pad and row
  /// \param pad ungrouped pad
  /// \param lrow local ungrouped row in a region
  unsigned int getGroupedPad(const unsigned int pad, const unsigned int ulrow) const { return getGroupedPad(pad, ulrow, mRegion, mGroupPads, mGroupRows, mRows, mGroupPadsSectorEdges, mPadsPerRow); };

  /// \return returns index to the data
  /// \param row local row of the grouped IDCs
  /// \param pad pad of the grouped IDCs
  /// \param integrationInterval integration interval
  unsigned int getIndex(const unsigned int glrow, const unsigned int pad, unsigned int integrationInterval) const { return mNIDCsPerCRU * integrationInterval + mOffsRow[glrow] + pad; }

  /// \return returns index to the data
  /// \param ulrow local ungrouped row
  /// \param upad ungrouped pad
  /// \param integrationInterval integration interval
  unsigned int getIndexUngrouped(const unsigned int ulrow, const unsigned int upad, unsigned int integrationInterval) const { return getIndex(getGroupedRow(ulrow), getGroupedPad(upad, ulrow), integrationInterval) + getOffsetForEdgePad(upad, ulrow); }

  /// \return returns offset of the index for a pad which is not grouped (in the region where mGroupPadsSectorEdges is true).
  /// returns for pads near local pad number = 0 negative value and for pads near local pad number = max value positive value
  /// \param upad ungrouped pad
  /// \param ulrow local ungrouped row
  /// \param groupRows grouping parameter for number of pads in row direction which are grouped
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are not grouped
  /// \param region region
  /// \param lastRow last ungrouped row
  static int getOffsetForEdgePad(const unsigned int upad, const unsigned int ulrow, const unsigned int groupRows, const unsigned int groupPadsSectorEdges, const unsigned int region, const int lastRow);

  /// \return returns offset of the index for a pad which is not grouped (in the region where mGroupPadsSectorEdges is true).
  /// returns for pads near local pad number = 0 negative value and for pads near local pad number = max value positive value
  /// \param upad ungrouped pad
  /// \param ulrow local ungrouped row
  int getOffsetForEdgePad(const unsigned int upad, const unsigned int ulrow) const { return (isSectorEdgePad(upad, ulrow, mRegion, mGroupPadsSectorEdges)) ? getOffsetForEdgePad(upad, ulrow, mGroupRows, mGroupPadsSectorEdges, mRegion, getLastRow()) : 0; }

  /// check if an ungrouped pad is a pad which will be grouped differently (i.e. is defined by mGroupPadsSectorEdges)
  /// \param upad ungrouped pad
  /// \param ulrow ungrouped local row
  /// \param region region
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are not grouped
  static bool isSectorEdgePad(const unsigned int upad, const unsigned int ulrow, const unsigned int region, const unsigned int groupPadsSectorEdges);

  /// \return returns index to the data
  /// \param ugrow global ungrouped row
  /// \param upad ungrouped pad
  /// \param integrationInterval integration interval
  unsigned int getIndexUngroupedGlob(const unsigned int ugrow, const unsigned int upad, unsigned int integrationInterval) const;

  /// \return returns the global pad number for given local pad row and pad
  /// \param ulrow local ungrouped row in a region
  /// \param pad ungrouped pad in row
  unsigned int getGlobalPadNumber(const unsigned int ulrow, const unsigned int pad) const;

  /// \return returns last ungrouped row
  unsigned int getLastRow() const;

  /// \return returns last ungrouped pad for given local row
  /// \param row local ungrouped row
  unsigned int getLastPad(const unsigned int ulrow) const { return getLastPad(ulrow, mRegion, mGroupPads, mGroupLastPadsThreshold, mGroupPadsSectorEdges); }

  /// \return returns last ungrouped pad for given local row
  /// \param ulrow local ungrouped row
  /// \param region region
  /// \param groupPads grouping parameter for number of pads in pad direction which are grouped
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are not grouped
  static unsigned int getLastPad(const unsigned int ulrow, const unsigned int region, const unsigned char groupPads, const unsigned char groupLastPadsThreshold, const unsigned int groupPadsSectorEdges);

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "IDCGroupHelperRegion.root", const char* outName = "IDCGroupHelperRegion") const;

  /// \return returns the number of differently grouped pads per row: returns sum of digits in integer whoch are not 0 (example: 0: returns 0, 11: returns 2, 321: returns 3)
  unsigned int getGroupedPadsSectorEdges() const;

  /// \return returns total number of pads which are grouped separately: calculate sum of all digits in integer value (example: 0: returns 0, 11: returns 2, 321: returns 6)
  unsigned int getTotalGroupPadsSectorEdges() const;

  /// \return return the index in the groups at the sector edge (example groupPadsSectorEdges=321: upad=0 returns 0, upad=1 returns 1, upad=2 returns 1, upad=3 returns 2, upad=5 returns 2)
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are not grouped
  /// \param urelpad ungrouped relative pad at the sector edge (pad at the sector edge starting from 0 and increases towards sector center)
  static unsigned int getIndexGroupPadsSectorEdges(const unsigned int groupPadsSectorEdges, const unsigned int urelpad);

  /// \return returns which type of grouping is performed for the sector edge pads
  EdgePadGroupingMethod getEdgePadGroupingType() const;

  /// \return returns number of ungrouped pads for grouped pad (example groupPadsSectorEdges=324530: group=0 -> 3, group=1 -> 5, group=2 -> 4...
  /// \param group index of the group
  unsigned int getPadsInGroupSectorEdges(const unsigned indexGroup) const;

 protected:
  const unsigned char mGroupPads{};              ///< grouping parameter in pad direction (how many pads in pad direction are grouped)
  const unsigned char mGroupRows{};              ///< grouping parameter in pad direction (how many pads in pad direction are grouped)
  const unsigned char mGroupLastRowsThreshold{}; ///< if the last group (region edges) consists in row direction less then mGroupLastRowsThreshold pads then it will be grouped into the previous group
  const unsigned char mGroupLastPadsThreshold{}; ///< if the last group (sector edges) consists in pad direction less then mGroupLastPadsThreshold pads then it will be grouped into the previous group
  const unsigned int mGroupPadsSectorEdges{};    ///< decoded number of pads at the sector edges which are grouped differently (example: 0: no pads are grouped, 11: first two pads are not grouped, 321: first pad is not grouped, second + third pads are grouped, fourth + fifth + sixth pads are grouped)
  const unsigned int mRegion{};                  ///< region of input IDCs
  unsigned int mNIDCsPerCRU{};                   ///< total number of IDCs per CRU per integration interval
  unsigned int mRows{};                          ///< number of grouped rows
  std::vector<unsigned int> mPadsPerRow{};       ///< number of grouped pads per grouped row
  std::vector<unsigned int> mOffsRow{};          ///< offset to calculate the index in the data from grouped row and grouped pad

  /// set number of grouped rows
  void setRows(const unsigned int nRows);

  /// initialize members
  void initIDCGroupHelperRegion();

  /// return the index in the groups at the sector edge (example groupPadsSectorEdges=321: upad=0 returns 0, upad=1 returns 1, upad=2 returns 1, upad=3 returns 2, upad=5 returns 2)
  /// \param urelpad ungrouped relative pad at the sector edge (pad at the sector edge starting from 0 and increases towards sector center)
  unsigned int getIndexGroupPadsSectorEdges(const unsigned int urelpad);

  /// helper function for getIndexGroupPadsSectorEdges(unsigned int groupPadsSectorEdges, const unsigned int urelpad)
  static unsigned int getIndexGroupPadsSectorEdges(const unsigned int groupPadsSectorEdges, const unsigned int urelpad, const unsigned int count);

  ClassDefNV(IDCGroupHelperRegion, 1)
};

} // namespace o2::tpc

#endif
