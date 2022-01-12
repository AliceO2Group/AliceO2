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
#include "TFile.h"
#include <numeric>

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
  return (upad < groupPadsSectorEdges || (upad >= Mapper::PADSPERROW[region][ulrow] - groupPadsSectorEdges)) ? true : false;
}

int o2::tpc::IDCGroupHelperRegion::getOffsetForEdgePad(const unsigned int upad, const unsigned int ulrow, const unsigned int groupRows, const unsigned int groupPadsSectorEdges, const unsigned int region, const int lastRow)
{
  if (ulrow < lastRow) {
    const int relPadinRow = ulrow % groupRows;
    const int localPadOffset = relPadinRow * groupPadsSectorEdges; // offset from pads to the left
    const int totalOff = upad < Mapper::PADSPERROW[region][ulrow] / 2 ? (localPadOffset + upad - groupPadsSectorEdges * groupRows) : (localPadOffset + upad - (Mapper::PADSPERROW[region][ulrow] - groupPadsSectorEdges) + 1);
    return totalOff;
  } else {
    const int relPadinRow = ulrow - lastRow;
    const int localPadOffset = relPadinRow * groupPadsSectorEdges; // offset from pads to the left
    const int totalOff = upad < Mapper::PADSPERROW[region][ulrow] / 2 ? (localPadOffset + upad - groupPadsSectorEdges * (Mapper::ROWSPERREGION[region] - lastRow)) : (localPadOffset + upad - (Mapper::PADSPERROW[region][ulrow] - groupPadsSectorEdges) + 1);
    return totalOff;
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

unsigned int o2::tpc::IDCGroupHelperRegion::getLastPad(const unsigned int ulrow) const
{
  const unsigned int nPads = Mapper::PADSPERROW[mRegion][ulrow] / 2 - mGroupPadsSectorEdges;
  const unsigned int padsRemainder = nPads % mGroupPads;
  int unsigned lastPad = (padsRemainder == 0) ? nPads - mGroupPads : nPads - padsRemainder;
  if (padsRemainder && padsRemainder <= mGroupLastPadsThreshold) {
    lastPad -= mGroupPads;
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

  mNIDCsPerCRU = std::accumulate(mPadsPerRow.begin(), mPadsPerRow.end(), decltype(mPadsPerRow)::value_type(0)) + 2 * Mapper::ROWSPERREGION[mRegion] * mGroupPadsSectorEdges;
  mOffsRow.front() = mGroupPadsSectorEdges * mGroupRows;
  for (unsigned int i = 1; i < (mRows - 1); ++i) {
    const unsigned int lastInd = i - 1;
    mOffsRow[i] = mOffsRow[lastInd] + mPadsPerRow[lastInd] + 2 * mGroupPadsSectorEdges * mGroupRows;
  }
  mOffsRow.back() = mOffsRow[mRows - 2] + mPadsPerRow[mRows - 2] + mGroupPadsSectorEdges * (mGroupRows + (Mapper::ROWSPERREGION[mRegion] - getLastRow()));
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
