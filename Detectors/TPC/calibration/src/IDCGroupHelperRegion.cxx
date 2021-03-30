// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

unsigned int o2::tpc::IDCGroupHelperRegion::getGroupedPad(const unsigned int upad, const unsigned int ulrow, const unsigned int region, const unsigned int groupPads, const unsigned int groupRows, const unsigned int groupedrows, const std::vector<unsigned int>& padsPerRow)
{
  const int relPadHalf = static_cast<int>(std::floor((upad - 0.5f * Mapper::PADSPERROW[region][ulrow]) / groupPads));
  const unsigned int nGroupedPads = padsPerRow[getGroupedRow(ulrow, groupRows, groupedrows)];
  const unsigned int nGroupedPadsHalf = (nGroupedPads / 2);
  if (std::abs(relPadHalf) >= nGroupedPadsHalf) {
    return std::signbit(relPadHalf) ? 0 : nGroupedPads - 1;
  }
  return static_cast<unsigned int>(static_cast<int>(nGroupedPadsHalf) + relPadHalf);
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
  const unsigned int nPads = Mapper::PADSPERROW[mRegion][ulrow] / 2;
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

  mNIDCsPerCRU = std::accumulate(mPadsPerRow.begin(), mPadsPerRow.end(), decltype(mPadsPerRow)::value_type(0));
  for (unsigned int i = 1; i < mRows; ++i) {
    const unsigned int lastInd = i - 1;
    mOffsRow[i] = mOffsRow[lastInd] + mPadsPerRow[lastInd];
  }
}

void o2::tpc::IDCGroupHelperRegion::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "UPDATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}
