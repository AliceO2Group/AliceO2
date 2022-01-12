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

#include "TPCCalibration/IDCAverageGroupHelper.h"
#include "TPCCalibration/IDCAverageGroupBase.h"
#include "TPCCalibration/IDCContainer.h"
#include "TPCCalibration/RobustAverage.h"

void o2::tpc::IDCAverageGroupHelper<o2::tpc::IDCAverageGroupCRU>::setGroupedIDC(const unsigned int rowGrouped, const unsigned int padGrouped)
{
  const static auto& paramIDCGroup = ParameterIDCGroup::Instance();
  switch (paramIDCGroup.method) {
    case o2::tpc::AveragingMethod::SLOW:
    default:
      mIDCsGrouped(rowGrouped, padGrouped, mIntegrationInterval) = mRobustAverage[mThreadNum].getFilteredAverage(paramIDCGroup.sigma);
      break;
    case o2::tpc::AveragingMethod::FAST:
      const float mean = mRobustAverage[mThreadNum].getMean();
      mIDCsGrouped(rowGrouped, padGrouped, mIntegrationInterval) = mRobustAverage[mThreadNum].getMean();
      break;
  }
}

void o2::tpc::IDCAverageGroupHelper<o2::tpc::IDCAverageGroupCRU>::set(const unsigned int threadNum, const unsigned int integrationInterval)
{
  mThreadNum = threadNum;
  mIntegrationInterval = integrationInterval;
  mOffsetUngrouped = integrationInterval * Mapper::PADSPERREGION[getRegion()];
}

void o2::tpc::IDCAverageGroupHelper<o2::tpc::IDCAverageGroupCRU>::addValue(const unsigned int padInRegion, const float weight)
{
  mRobustAverage[mThreadNum].addValue(getUngroupedIDCVal(padInRegion) * Mapper::INVPADAREA[getRegion()], weight);
}

void o2::tpc::IDCAverageGroupHelper<o2::tpc::IDCAverageGroupCRU>::clearRobustAverage()
{
  mRobustAverage[mThreadNum].clear();
}

float o2::tpc::IDCAverageGroupHelper<o2::tpc::IDCAverageGroupTPC>::getGroupedIDCValGlobal(unsigned int urow, unsigned int upad) const
{
  return mIDCsGrouped.getValue(getSide(), mIDCGroupHelperSector.getIndexUngroupedGlobal(getSector(), getRegion(), urow, upad, mIntegrationInterval));
}

void o2::tpc::IDCAverageGroupHelper<o2::tpc::IDCAverageGroupTPC>::setGroupedIDC(const unsigned int glrow, const unsigned int padGrouped, const float val)
{
  const unsigned int index = mIDCGroupHelperSector.getOffsRow(getRegion(), glrow) + padGrouped + mOffsetGrouped;
  mIDCsGrouped.setValue(val, getSide(), index);
}

float o2::tpc::IDCAverageGroupHelper<o2::tpc::IDCAverageGroupTPC>::getUngroupedIDCVal(const unsigned int padInRegion) const
{
  return mIDCsUngrouped.getValue(getSide(), padInRegion + mOffsetUngrouped);
}

void o2::tpc::IDCAverageGroupHelper<o2::tpc::IDCAverageGroupTPC>::setIntegrationInterval(const unsigned int integrationInterval)
{
  mIntegrationInterval = integrationInterval;
  mOffsetUngrouped = (mIntegrationInterval * SECTORSPERSIDE + getSector() % SECTORSPERSIDE) * Mapper::getPadsInSector() + Mapper::GLOBALPADOFFSET[getRegion()];
  mOffsetGrouped = mIDCGroupHelperSector.getIndexGrouped(getSector(), getRegion(), 0, 0, mIntegrationInterval) - mIDCGroupHelperSector.getGroupingParameter().getGroupPadsSectorEdges() * mIDCGroupHelperSector.getGroupingParameter().getGroupRows(getRegion());
}

void o2::tpc::IDCAverageGroupHelper<o2::tpc::IDCAverageGroupTPC>::setGroupedIDC(const unsigned int rowGrouped, const unsigned int padGrouped)
{
  const static auto& paramIDCGroup = ParameterIDCGroup::Instance();
  switch (paramIDCGroup.method) {
    case o2::tpc::AveragingMethod::SLOW:
    default:
      setGroupedIDC(rowGrouped, padGrouped, mRobustAverage[mThreadNum].getFilteredAverage(paramIDCGroup.sigma));
      break;
    case o2::tpc::AveragingMethod::FAST:
      setGroupedIDC(rowGrouped, padGrouped, mRobustAverage[mThreadNum].getMean());
      break;
  }
}

void o2::tpc::IDCAverageGroupHelper<o2::tpc::IDCAverageGroupTPC>::addValue(const unsigned int padInRegion, const float weight)
{
  mRobustAverage[mThreadNum].addValue(getUngroupedIDCVal(padInRegion), weight);
}

void o2::tpc::IDCAverageGroupHelper<o2::tpc::IDCAverageGroupTPC>::clearRobustAverage()
{
  mRobustAverage[mThreadNum].clear();
}

void o2::tpc::IDCAverageGroupHelper<o2::tpc::IDCAverageGroupTPC>::setSectorEdgeIDC(const unsigned int ulrow, const unsigned int upad, const unsigned int padInRegion)
{
  const int index = mIDCGroupHelperSector.getIndexUngrouped(getSector(), getRegion(), ulrow, upad, mIntegrationInterval);
  mIDCsGrouped.setValue(getUngroupedIDCVal(padInRegion), getSide(), index);
}
