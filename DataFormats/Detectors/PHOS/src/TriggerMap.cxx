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

#include "PHOSBase/Geometry.h"
#include "DataFormatsPHOS/TriggerMap.h"

#include <fairlogger/Logger.h>

#include <TH2.h>
#include <TRandom.h>

#include <iostream>

using namespace o2::phos;

TriggerMap::TriggerMap(int param) : mVersion(param)
{
  // create default object
  // empty (all channels good) bad maps and
  // uniform turn-on curves for DDLs
  mParamDescr.emplace_back("TestDefault");
  std::array<std::array<float, NMAXPAR>, NDDL> a;
  for (int iDDL = 0; iDDL < NDDL; iDDL++) {
    a[iDDL].fill(0);
    a[iDDL][0] = 1.;  // only one step
    a[iDDL][1] = 4.;  // threshold
    a[iDDL][2] = 0.5; // width
  }
  mParamSets.emplace_back(a);
  mCurrentSet = mParamSets[0];
  mVersion = 0;
}

void TriggerMap::addTurnOnCurvesParams(std::string_view versionName, std::array<std::array<float, 10>, 14>& params)
{
  mParamDescr.emplace_back(versionName);
  mParamSets.emplace_back(params);
}

void TriggerMap::setTurnOnCurvesVestion(int v)
{
  if (static_cast<std::size_t>(v) >= mParamDescr.size()) {
    LOG(error) << "impossible parameterization " << v;
    LOG(error) << "Available are:";
    for (std::size_t i = 0; i < mParamDescr.size(); i++) {
      LOG(error) << i << " : " << mParamDescr[i];
    }
    LOG(error) << " keep current " << mParamDescr[mVersion];
    return;
  }
  mVersion = v;
  LOG(info) << "Will use parameterization " << mParamDescr[mVersion];
  mCurrentSet = mParamSets[mVersion];
}

bool TriggerMap::selectTurnOnCurvesParams(std::string_view versionName)
{
  mVersion = 0;
  while (static_cast<std::size_t>(mVersion) < mParamDescr.size()) {
    if (versionName.compare(mParamDescr[mVersion]) == 0.) {
      return true;
    }
    mVersion++;
  }
  mVersion = 0;
  LOG(error) << "Can not fine parameterization " << versionName;
  LOG(error) << "Available are:";
  for (std::size_t i = 0; i < mParamDescr.size(); i++) {
    LOG(error) << i << " : " << mParamDescr[i];
  }
  return false;
}

float TriggerMap::L0triggerProbability(float e, short ddl) const
{

  if (mCurrentSet.size() == 0) {
    LOG(error) << "Parameteriztion not chosen";
    return 0;
  }
  if (mVersion == 0) {
    return mCurrentSet[ddl][0] / (TMath::Exp((mCurrentSet[ddl][1] - e) / mCurrentSet[ddl][2]) + 1.) +
           (1. - mCurrentSet[ddl][0]) / (TMath::Exp((mCurrentSet[ddl][3] - e) / mCurrentSet[ddl][4]) + 1.);
  } else {
    return 0;
  }
}
bool TriggerMap::isFiredMC2x2(float a, short module, short ix, short iz) const
{
  char truRelId[3] = {char(module), char(ix), char(iz)};
  short tileId = Geometry::truRelToAbsNumbering(truRelId, 0); // 0 for 2x2 TriggerMap
  short iTRU = module * 4 + ix / 16 - 6;
  return isGood2x2(tileId) && try2x2(a, iTRU);
}

bool TriggerMap::isFiredMC4x4(float a, short module, short ix, short iz) const
{
  char truRelId[3] = {char(module), char(ix), char(iz)};
  short tileId = Geometry::truRelToAbsNumbering(truRelId, 1);
  short iTRU = module * 4 + ix / 16 - 6;
  return isGood4x4(tileId) && try4x4(a, iTRU);
}
bool TriggerMap::try2x2(float a, short iTRU) const
{
  return gRandom->Uniform() < L0triggerProbability(a, iTRU);
}
bool TriggerMap::try4x4(float a, short iTRU) const
{
  return gRandom->Uniform() < L0triggerProbability(a, iTRU);
}
