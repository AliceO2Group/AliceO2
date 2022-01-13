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

/// \file IDCGroupHelperSector.h
/// \brief helper class for grouping of pads and rows for one sector
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_IDCGROUPHELPERSECTOR_H_
#define ALICEO2_TPC_IDCGROUPHELPERSECTOR_H_

#include <vector>
#include "Rtypes.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/IDCGroupingParameter.h"

namespace o2::tpc
{

/// Helper class for accessing grouped pads for one sector
class IDCGroupHelperSector
{
 public:
  /// constructor
  /// \param groupPads number of pads in pad direction which will be grouped
  /// \param groupRows number of pads in row direction which will be grouped
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  IDCGroupHelperSector(const std::array<unsigned char, Mapper::NREGIONS>& groupPads, const std::array<unsigned char, Mapper::NREGIONS>& groupRows, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold, const unsigned int groupNotnPadsSectorEdges)
    : mGroupingPar{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupNotnPadsSectorEdges} { initIDCGroupHelperSector(); };

  /// constructor
  /// \param groupingParameter struct holding the grouping parameter
  IDCGroupHelperSector(const ParameterIDCGroupCCDB& groupingParameter) : mGroupingPar{groupingParameter} { initIDCGroupHelperSector(); };

  /// default constructor for ROOT I/O
  IDCGroupHelperSector() = default;

  /// \return returns index to the data
  /// \param sector sector
  /// \param region TPC region
  /// \param glrow grouped local row
  /// \param pad pad of the grouped IDCs
  /// \param integrationInterval integration interval
  unsigned int getIndexGrouped(const unsigned int sector, const unsigned int region, const unsigned int glrow, const unsigned int pad, unsigned int integrationInterval) const { return mNIDCsPerSector * (integrationInterval * SECTORSPERSIDE + sector % o2::tpc::SECTORSPERSIDE) + mRegionOffs[region] + mOffsRow[region][glrow] + pad; }

  /// \return returns the index to the grouped data with ungrouped inputs
  /// \param sector sector
  /// \param region TPC region
  /// \param ulrow local row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  unsigned int getIndexUngrouped(const unsigned int sector, const unsigned int region, unsigned int ulrow, unsigned int upad, unsigned int integrationInterval) const { return getIndexGrouped(sector, region, getGroupedRow(region, ulrow), getGroupedPad(region, ulrow, upad), integrationInterval) + getOffsetForEdgePad(upad, ulrow, region); }

  /// \return returns offset of the index for a pad which is not grouped (in the region where mGroupPadsSectorEdges is true).
  /// returns for pads near local pad number = 0 negative value and for pads near local pad number = max value positive value
  /// \param upad pad number of the ungrouped IDCs
  /// \param ulrow local row of the ungrouped IDCs
  /// \param region TPC region
  int getOffsetForEdgePad(const unsigned int upad, const unsigned int ulrow, const unsigned int region) const;

  /// \return returns the index to the grouped data with ungrouped inputs
  /// \param sector sector
  /// \param region TPC region
  /// \param ugrow global row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  unsigned int getIndexUngroupedGlobal(const unsigned int sector, const unsigned int region, unsigned int ugrow, unsigned int upad, unsigned int integrationInterval) const { return getIndexUngrouped(sector, region, ugrow - Mapper::ROWOFFSET[region], upad, integrationInterval); }

  /// \return returns grouped pad for ungrouped row and pad
  /// \param region region
  /// \param ulrow local ungrouped row in a region
  /// \param upad ungrouped pad
  unsigned int getGroupedPad(const unsigned int region, unsigned int ulrow, unsigned int upad) const;

  /// \return returns the row of the group from the local ungrouped row in a region
  /// \param region region
  /// \param ulrow local ungrouped row in a region
  unsigned int getGroupedRow(const unsigned int region, unsigned int ulrow) const;

  /// \returns grouping parameter
  const auto& getGroupingParameter() const { return mGroupingPar; }

  /// \return returns number of IDCs for given region
  unsigned int getNIDCs(const unsigned int region) const { return mNIDCsPerCRU[region]; }

  /// \return returns number of grouped rows for given region
  unsigned int getNRows(const unsigned int region) const { return mRows[region]; }

  /// \return returns region offset for calculating the index
  unsigned int getRegionOffset(const unsigned int region) const { return mRegionOffs[region]; }

  /// \return returns number of IDCs for a whole sector
  unsigned int getNIDCsPerSector() const { return mNIDCsPerSector; }

  /// \return returns last ungrouped row
  unsigned int getLastRow(const unsigned int region) const;

  /// \return returns last ungrouped pad for given global row
  /// \param ulrow ungrouped local row
  unsigned int getLastPad(const unsigned int region, const unsigned int ulrow) const;

  /// \return returns offsey to calculate the index
  /// \param glrow grouped local row
  unsigned int getOffsRow(const unsigned int region, const unsigned int glrow) const { return mOffsRow[region][glrow]; }

  /// \return returns number of grouped pads per row
  /// \param glrow grouped local row
  unsigned getPadsPerRow(const unsigned int region, const unsigned int glrow) const { return mPadsPerRow[region][glrow]; }

  /// \return returns index to ungrouped data from ungrouped pad and row
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  static unsigned int getUngroupedIndexGlobal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) { return (integrationInterval * SECTORSPERSIDE + sector % SECTORSPERSIDE) * Mapper::getPadsInSector() + Mapper::GLOBALPADOFFSET[region] + Mapper::OFFSETCRULOCAL[region][urow] + upad; }

 protected:
  ParameterIDCGroupCCDB mGroupingPar{};                                  ///< struct containg the grouping parameter
  std::array<unsigned int, Mapper::NREGIONS> mNIDCsPerCRU{};             ///< total number of IDCs per region per integration interval
  unsigned int mNIDCsPerSector{};                                        ///< number of grouped IDCs per sector
  std::array<unsigned int, Mapper::NREGIONS> mRows{};                    ///< number of grouped rows per region
  std::array<unsigned int, Mapper::NREGIONS> mRegionOffs{};              ///< offset for the region per region
  std::array<std::vector<unsigned int>, Mapper::NREGIONS> mPadsPerRow{}; ///< number of pads per row per region
  std::array<std::vector<unsigned int>, Mapper::NREGIONS> mOffsRow{};    ///< offset to calculate the index in the data from row and pad per region

  /// init function for setting the members
  void initIDCGroupHelperSector();

  ClassDefNV(IDCGroupHelperSector, 1)
};

} // namespace o2::tpc

#endif
