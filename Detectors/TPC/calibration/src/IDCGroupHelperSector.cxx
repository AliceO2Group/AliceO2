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

#include "TPCCalibration/IDCGroupHelperSector.h"
#include "TPCCalibration/IDCGroupHelperRegion.h"
#include <numeric>

unsigned int o2::tpc::IDCGroupHelperSector::getGroupedPad(const unsigned int region, unsigned int ulrow, unsigned int upad) const
{
  return IDCGroupHelperRegion::getGroupedPad(upad, ulrow, region, mGroupingPar.GroupPads[region], mGroupingPar.GroupRows[region], mRows[region], mPadsPerRow[region]);
}

unsigned int o2::tpc::IDCGroupHelperSector::getGroupedRow(const unsigned int region, unsigned int ulrow) const
{
  return IDCGroupHelperRegion::getGroupedRow(ulrow, mGroupingPar.GroupRows[region], mRows[region]);
}

unsigned int o2::tpc::IDCGroupHelperSector::getLastRow(const unsigned int region) const
{
  const unsigned int nTotRows = Mapper::ROWSPERREGION[region];
  const unsigned int rowsRemainder = nTotRows % mGroupingPar.GroupRows[region];
  unsigned int lastRow = nTotRows - rowsRemainder;
  if (rowsRemainder <= mGroupingPar.GroupLastRowsThreshold[region]) {
    lastRow -= mGroupingPar.GroupRows[region];
  }
  return lastRow;
}

unsigned int o2::tpc::IDCGroupHelperSector::getLastPad(const unsigned int region, const unsigned int ulrow) const
{
  const unsigned int nPads = Mapper::PADSPERROW[region][ulrow] / 2;
  const unsigned int padsRemainder = nPads % mGroupingPar.GroupPads[region];
  int unsigned lastPad = (padsRemainder == 0) ? nPads - mGroupingPar.GroupPads[region] : nPads - padsRemainder;
  if (padsRemainder && padsRemainder <= mGroupingPar.GroupLastPadsThreshold[region]) {
    lastPad -= mGroupingPar.GroupPads[region];
  }
  return lastPad;
}

void o2::tpc::IDCGroupHelperSector::initIDCGroupHelperSector()
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
