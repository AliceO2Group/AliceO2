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

#include "TPCCalibration/IDCGroupHelperRegion.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/IDCGroupingParameter.h"
#include "TFile.h"
#include <numeric>
#include <fmt/format.h>

unsigned int o2::tpc::IDCGroupHelperRegion::getGlobalPadNumber(const unsigned int ulrow, const unsigned int pad) const
{
  return Mapper::getGlobalPadNumber(ulrow, pad, mRegion);
}

unsigned int o2::tpc::IDCGroupHelperRegion::getGroupedRow(const unsigned int ulrow, const unsigned int groupRows, const unsigned int groupedrows)
{
  const unsigned int row = ulrow / groupRows;
  return (row >= groupedrows) ? (groupedrows - 1) : row;
}

unsigned int o2::tpc::IDCGroupHelperRegion::getGroupedPad(const unsigned int upad, const unsigned int ulrow, const unsigned int region, const unsigned int groupPads, const unsigned int groupRows, const unsigned int groupedrows, const unsigned int groupPadsSectorEdges, const std::vector<unsigned int>& padsPerRow)
{
  const int relPadHalf = static_cast<int>(std::floor((upad - 0.5f * Mapper::PADSPERROW[region][ulrow]) / groupPads));
  const unsigned int nGroupedPads = padsPerRow[getGroupedRow(ulrow, groupRows, groupedrows)];
  const unsigned int nGroupedPadsHalf = (nGroupedPads / 2);
  if (std::abs(relPadHalf) >= nGroupedPadsHalf) {
    return std::signbit(relPadHalf) ? 0 : nGroupedPads - 1;
  }
  return static_cast<unsigned int>(static_cast<int>(nGroupedPadsHalf) + relPadHalf);
}

bool o2::tpc::IDCGroupHelperRegion::isSectorEdgePad(const unsigned int upad, const unsigned int ulrow, const unsigned int region, const unsigned int groupPadsSectorEdges)
{
  return (upad < ParameterIDCGroup::getTotalGroupPadsSectorEdges(groupPadsSectorEdges) || (upad >= Mapper::PADSPERROW[region][ulrow] - ParameterIDCGroup::getTotalGroupPadsSectorEdges(groupPadsSectorEdges))) ? true : false;
}

int o2::tpc::IDCGroupHelperRegion::getOffsetForEdgePad(const unsigned int upad, const unsigned int ulrow, const unsigned int groupRows, const unsigned int groupPadsSectorEdges, const unsigned int region, const int lastRow)
{
  const auto edgePadGroupingType = ParameterIDCGroup::getEdgePadGroupingType(groupPadsSectorEdges);
  const auto padsGroupedSecEdge = ParameterIDCGroup::getGroupedPadsSectorEdges(groupPadsSectorEdges);
  const bool isLeftSide = upad < Mapper::PADSPERROW[region][ulrow] / 2;
  if (edgePadGroupingType == EdgePadGroupingMethod::NO) {
    if (ulrow < lastRow) {
      const int relPadinRow = ulrow % groupRows;
      const int localPadOffset = relPadinRow * padsGroupedSecEdge; // offset from pads to the left
      const int totalOff = isLeftSide ? (localPadOffset + getIndexGroupPadsSectorEdges(groupPadsSectorEdges, upad) - padsGroupedSecEdge * groupRows) : (localPadOffset + getIndexGroupPadsSectorEdges(groupPadsSectorEdges, Mapper::PADSPERROW[region][ulrow] - upad - 1) + 1);
      return totalOff;
    } else {
      const int relPadinRow = ulrow - lastRow;
      const int localPadOffset = relPadinRow * padsGroupedSecEdge; // offset from pads to the left
      const int totalOff = isLeftSide ? (localPadOffset + getIndexGroupPadsSectorEdges(groupPadsSectorEdges, upad) - padsGroupedSecEdge * (Mapper::ROWSPERREGION[region] - lastRow)) : (localPadOffset + getIndexGroupPadsSectorEdges(groupPadsSectorEdges, Mapper::PADSPERROW[region][ulrow] - upad - 1) + 1);
      return totalOff;
    }
  } else if (edgePadGroupingType == EdgePadGroupingMethod::ROWS) {
    const int totalOff = isLeftSide ? (getIndexGroupPadsSectorEdges(groupPadsSectorEdges, upad) - padsGroupedSecEdge) : (getIndexGroupPadsSectorEdges(groupPadsSectorEdges, Mapper::PADSPERROW[region][ulrow] - upad - 1) + 1);
    return totalOff;
  } else {
    // wrong type
    throw std::invalid_argument("Wrong type for EdgePadGroupingMethod");
  }
}

void o2::tpc::IDCGroupHelperRegion::setRows(const unsigned int nRows)
{
  mRows = nRows;
  mPadsPerRow.resize(mRows);
  mOffsRow.resize(mRows);
}

unsigned int o2::tpc::IDCGroupHelperRegion::getLastRow() const
{
  const unsigned int nTotRows = Mapper::ROWSPERREGION[mRegion];
  const unsigned int rowsRemainder = nTotRows % mGroupRows;
  unsigned int lastRow = nTotRows - rowsRemainder;
  if (rowsRemainder <= mGroupLastRowsThreshold) {
    lastRow -= mGroupRows;
  }
  return lastRow;
}

unsigned int o2::tpc::IDCGroupHelperRegion::getLastPad(const unsigned int ulrow, const unsigned int region, const unsigned char groupPads, const unsigned char groupLastPadsThreshold, const unsigned int groupPadsSectorEdges)
{
  const unsigned int nPads = Mapper::PADSPERROW[region][ulrow] / 2 - ParameterIDCGroup::getTotalGroupPadsSectorEdges(groupPadsSectorEdges);
  const unsigned int padsRemainder = nPads % groupPads;
  int unsigned lastPad = (padsRemainder == 0) ? nPads - groupPads : nPads - padsRemainder;
  if (padsRemainder && padsRemainder <= groupLastPadsThreshold) {
    lastPad -= groupPads;
  }
  return lastPad;
}

void o2::tpc::IDCGroupHelperRegion::initIDCGroupHelperRegion()
{
  const unsigned int nRows = getLastRow() / mGroupRows + 1;
  setRows(nRows);
  for (unsigned int irow = 0; irow < nRows; ++irow) {
    const unsigned int row = irow * mGroupRows;
    mPadsPerRow[irow] = 2 * (getLastPad(row) / mGroupPads + 1);
  }

  const auto padsSectorEdge = getEdgePadGroupingType() == EdgePadGroupingMethod::NO ? Mapper::ROWSPERREGION[mRegion] : nRows;
  const auto rowsSectorEdge = getEdgePadGroupingType() == EdgePadGroupingMethod::NO ? mGroupRows : 1;

  mNIDCsPerCRU = std::accumulate(mPadsPerRow.begin(), mPadsPerRow.end(), decltype(mPadsPerRow)::value_type(0)) + 2 * padsSectorEdge * getGroupedPadsSectorEdges();
  mOffsRow.front() = getGroupedPadsSectorEdges() * rowsSectorEdge;
  for (unsigned int i = 1; i < (mRows - 1); ++i) {
    const unsigned int lastInd = i - 1;
    mOffsRow[i] = mOffsRow[lastInd] + mPadsPerRow[lastInd] + 2 * getGroupedPadsSectorEdges() * rowsSectorEdge;
  }
  if (mRows > 2) {
    const auto offsIndex = getEdgePadGroupingType() == EdgePadGroupingMethod::NO ? mGroupRows + (Mapper::ROWSPERREGION[mRegion] - getLastRow()) : 2;
    mOffsRow.back() = mOffsRow[mRows - 2] + mPadsPerRow[mRows - 2] + getGroupedPadsSectorEdges() * offsIndex;
  }
}

void o2::tpc::IDCGroupHelperRegion::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "UPDATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

unsigned int o2::tpc::IDCGroupHelperRegion::getIndexUngroupedGlob(const unsigned int ugrow, const unsigned int upad, unsigned int integrationInterval) const
{
  return getIndexUngrouped(ugrow - o2::tpc::Mapper::ROWOFFSET[mRegion], upad, integrationInterval);
}

unsigned int o2::tpc::IDCGroupHelperRegion::getTotalGroupPadsSectorEdges() const
{
  return ParameterIDCGroup::getTotalGroupPadsSectorEdges(mGroupPadsSectorEdges);
}

unsigned int o2::tpc::IDCGroupHelperRegion::getGroupedPadsSectorEdges() const
{
  return ParameterIDCGroup::getGroupedPadsSectorEdges(mGroupPadsSectorEdges);
}

unsigned int o2::tpc::IDCGroupHelperRegion::getIndexGroupPadsSectorEdges(const unsigned int urelpad)
{
  return getIndexGroupPadsSectorEdges(mGroupPadsSectorEdges, urelpad);
}

unsigned int o2::tpc::IDCGroupHelperRegion::getIndexGroupPadsSectorEdges(const unsigned int groupPadsSectorEdges, const unsigned int urelpad, const unsigned int count)
{
  const auto totalUngroupedPads = groupPadsSectorEdges % 10 + count;
  if (urelpad < totalUngroupedPads) {
    return 0;
  }
  return 1 + getIndexGroupPadsSectorEdges(groupPadsSectorEdges / 10, urelpad, totalUngroupedPads);
}

unsigned int o2::tpc::IDCGroupHelperRegion::getIndexGroupPadsSectorEdges(const unsigned int groupPadsSectorEdges, const unsigned int urelpad)
{
  return urelpad >= ParameterIDCGroup::getTotalGroupPadsSectorEdges(groupPadsSectorEdges) ? throw std::invalid_argument(fmt::format("relativ pad position {} is large than maximum value of {}", urelpad, ParameterIDCGroup::getTotalGroupPadsSectorEdges(groupPadsSectorEdges))) : getIndexGroupPadsSectorEdges(groupPadsSectorEdges / 10, urelpad, 0);
}

o2::tpc::EdgePadGroupingMethod o2::tpc::IDCGroupHelperRegion::getEdgePadGroupingType() const
{
  return ParameterIDCGroup::getEdgePadGroupingType(mGroupPadsSectorEdges);
}

unsigned int o2::tpc::IDCGroupHelperRegion::getPadsInGroupSectorEdges(const unsigned indexGroup) const
{
  return ParameterIDCGroup::getPadsInGroupSectorEdges(mGroupPadsSectorEdges, indexGroup);
}
