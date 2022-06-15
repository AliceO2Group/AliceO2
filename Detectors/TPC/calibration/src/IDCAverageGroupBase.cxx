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

#include "TPCCalibration/IDCAverageGroupBase.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/IDCDrawHelper.h"

void o2::tpc::IDCAverageGroupBase<o2::tpc::IDCAverageGroupCRU>::drawUngroupedIDCs(const unsigned int integrationInterval, const std::string filename) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this, integrationInterval](const unsigned int, const unsigned int region, const unsigned int row, const unsigned int pad) {
    return this->getUngroupedNormedIDCValLocal(row, pad, integrationInterval);
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDC);
  IDCDrawHelper::drawSector(drawFun, this->mIDCsGrouped.getRegion(), this->mIDCsGrouped.getRegion() + 1, 0, zAxisTitle, filename);
}

void o2::tpc::IDCAverageGroupBase<o2::tpc::IDCAverageGroupCRU>::setIDCs(const std::vector<float>& idcs)
{
  mIDCsUngrouped = idcs;
  mIDCsGrouped.resize(getNIntegrationIntervals());
}

void o2::tpc::IDCAverageGroupBase<o2::tpc::IDCAverageGroupCRU>::setIDCs(std::vector<float>&& idcs)
{
  mIDCsUngrouped = std::move(idcs);
  mIDCsGrouped.resize(getNIntegrationIntervals());
}

void o2::tpc::IDCAverageGroupBase<o2::tpc::IDCAverageGroupTPC>::drawIDCDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const bool grouped, const std::string filename) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this, integrationInterval, grouped](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return grouped ? this->getGroupedIDCDeltaVal(sector, region, irow, pad, integrationInterval) : this->getUngroupedIDCDeltaVal(sector, region, irow, pad, integrationInterval);
  };
  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDCDelta, IDCDeltaCompression::NO);
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename);
}

void o2::tpc::IDCAverageGroupBase<o2::tpc::IDCAverageGroupTPC>::resizeGroupedIDCs()
{
  const unsigned int nIDCs = mIDCGroupHelperSector.getNIDCsPerSector() * SECTORSPERSIDE * getNIntegrationIntervals();
  mIDCsGrouped.resize(nIDCs);
}

void o2::tpc::IDCAverageGroupBase<o2::tpc::IDCAverageGroupTPC>::setIDCs(const IDCDelta<float>& idcs, const Side side)
{
  mSide = side;
  mIDCsUngrouped = idcs;
  resizeGroupedIDCs();
}

void o2::tpc::IDCAverageGroupBase<o2::tpc::IDCAverageGroupTPC>::setIDCs(IDCDelta<float>&& idcs, const Side side)
{
  mSide = side;
  mIDCsUngrouped = std::move(idcs);
  resizeGroupedIDCs();
}

void o2::tpc::IDCAverageGroupBase<o2::tpc::IDCAverageGroupTPC>::resetGroupedIDCs()
{
  std::fill(mIDCsGrouped.getIDCDelta().begin(), mIDCsGrouped.getIDCDelta().end(), 0);
}
