// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <numeric>
#include "Rtypes.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/IDCGroupHelperRegion.h"
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
  IDCGroupHelperSector(const std::array<unsigned char, Mapper::NREGIONS>& groupPads, const std::array<unsigned char, Mapper::NREGIONS>& groupRows, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold)
    : mGroupingPar{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold} { initIDCGroupHelperSector(); };

  /// constructor
  /// \param groupingParameter struct holding the grouping parameter
  IDCGroupHelperSector(const ParameterIDCGroupCCDB& groupingParameter) : mGroupingPar{groupingParameter} { initIDCGroupHelperSector(); };

  /// default constructor for ROOT I/O
  IDCGroupHelperSector() = default;

  /// \return returns index to the data
  /// \param glrow grouped local row
  /// \param pad pad of the grouped IDCs
  unsigned int getIndexGrouped(const unsigned int sector, const unsigned int region, const unsigned int glrow, const unsigned int pad, unsigned int integrationInterval) const { return mNIDCsPerSector * (integrationInterval * SECTORSPERSIDE + sector) + mRegionOffs[region] + mOffsRow[region][glrow] + pad; }

  /// \return returns the index to the grouped data with ungrouped inputs
  /// \param sector sector
  /// \param region TPC region
  /// \param ulrow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  unsigned int getIndexUngrouped(const unsigned int sector, const unsigned int region, unsigned int ulrow, unsigned int upad, unsigned int integrationInterval) const { return getIndexGrouped(sector % o2::tpc::SECTORSPERSIDE, region, getGroupedRow(region, ulrow), getGroupedPad(region, ulrow, upad), integrationInterval); }

  /// \return returns grouped pad for ungrouped row and pad
  /// \param region region
  /// \param ulrow local ungrouped row in a region
  /// \param upad ungrouped pad
  unsigned int getGroupedPad(const unsigned int region, unsigned int ulrow, unsigned int upad) const { return IDCGroupHelperRegion::getGroupedPad(upad, ulrow, region, mGroupingPar.GroupPads[region], mGroupingPar.GroupRows[region], mRows[region], mPadsPerRow[region]); }

  /// \return returns the row of the group from the local ungrouped row in a region
  /// \param region region
  /// \param ulrow local ungrouped row in a region
  unsigned int getGroupedRow(const unsigned int region, unsigned int ulrow) const { return IDCGroupHelperRegion::getGroupedRow(ulrow, mGroupingPar.GroupRows[region], mRows[region]); }

  /// \returns grouping parameter
  const auto& getGroupingParameter() const { return mGroupingPar; }

  /// \return returns number if IDCs for given region
  unsigned int getNIDCs(const unsigned int region) { return mNIDCsPerCRU[region]; }

 protected:
  ParameterIDCGroupCCDB mGroupingPar{};                                  ///< struct containg the grouping parameter
  std::array<unsigned int, Mapper::NREGIONS> mNIDCsPerCRU{};             ///< total number of IDCs per region per integration interval
  unsigned int mNIDCsPerSector{};                                        ///< number of grouped IDCs per sector
  std::array<unsigned int, Mapper::NREGIONS> mRows{};                    ///< number of grouped rows per region
  std::array<unsigned int, Mapper::NREGIONS> mRegionOffs{};              ///< offset for the region per region
  std::array<std::vector<unsigned int>, Mapper::NREGIONS> mPadsPerRow{}; ///< number of pads per row per region
  std::array<std::vector<unsigned int>, Mapper::NREGIONS> mOffsRow{};    ///< offset to calculate the index in the data from row and pad per region

  void initIDCGroupHelperSector()
  {
    for (unsigned int reg = 0; reg < Mapper::NREGIONS; ++reg) {
      const IDCGroupHelperRegion groupTmp(mGroupingPar.GroupPads[reg], mGroupingPar.GroupRows[reg], mGroupingPar.GroupLastRowsThreshold[reg], mGroupingPar.GroupLastPadsThreshold[reg], reg);
      mNIDCsPerCRU[reg] = groupTmp.getNIDCsPerIntegrationInterval();
      mRows[reg] = groupTmp.getNRows();
      mPadsPerRow[reg] = groupTmp.getPadsPerRow();
      mOffsRow[reg] = groupTmp.getRowOffset();
      if (reg > 0) {
        const unsigned int lastInd = reg - 1;
        mRegionOffs[reg] = mRegionOffs[lastInd] + mNIDCsPerCRU[lastInd];
      }
    }
    mNIDCsPerSector = static_cast<unsigned int>(std::accumulate(mNIDCsPerCRU.begin(), mNIDCsPerCRU.end(), decltype(mNIDCsPerCRU)::value_type(0)));
  }

  ClassDefNV(IDCGroupHelperSector, 1)
};

} // namespace o2::tpc

#endif
