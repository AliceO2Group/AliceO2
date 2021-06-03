// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSBase/Geometry.h"
#include "DataFormatsPHOS/TriggerMap.h"

#include "FairLogger.h"

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
    a[iDDL][0] = 1.;  //only one step
    a[iDDL][1] = 4.;  //threshold
    a[iDDL][2] = 0.5; //width
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
  if (v >= mParamDescr.size()) {
    LOG(ERROR) << "impossible parameterization " << v;
    LOG(ERROR) << "Available are:";
    for (int i = 0; i < mParamDescr.size(); i++) {
      LOG(ERROR) << i << " : " << mParamDescr[i];
    }
    LOG(ERROR) << " keep current " << mParamDescr[mVersion];
    return;
  }
  mVersion = v;
  LOG(INFO) << "Will use parameterization " << mParamDescr[mVersion];
  mCurrentSet = mParamSets[mVersion];
}

bool TriggerMap::selectTurnOnCurvesParams(std::string_view versionName)
{
  mVersion = 0;
  while (mVersion < mParamDescr.size()) {
    if (versionName.compare(mParamDescr[mVersion]) == 0.) {
      return true;
    }
    mVersion++;
  }
  mVersion = 0;
  LOG(ERROR) << "Can not fine parameterization " << versionName;
  LOG(ERROR) << "Available are:";
  for (int i = 0; i < mParamDescr.size(); i++) {
    LOG(ERROR) << i << " : " << mParamDescr[i];
  }
  return false;
}

float TriggerMap::L0triggerProbability(float e, short ddl) const
{

  if (mCurrentSet.size() == 0) {
    LOG(ERROR) << "Parameteriztion not chosen";
    return 0;
  }
  if (mVersion == 0) {
    return mCurrentSet[ddl][0] / (TMath::Exp((mCurrentSet[ddl][1] - e) / mCurrentSet[ddl][2]) + 1.) +
           (1. - mCurrentSet[ddl][0]) / (TMath::Exp((mCurrentSet[ddl][3] - e) / mCurrentSet[ddl][4]) + 1.);
  } else {
    return 0;
  }
}
bool TriggerMap::isFiredMC2x2(float a, short iTRU, short ix, short iz) const
{
  char truRelId[3] = {char(iTRU), char(ix), char(iz)};
  short tileId = Geometry::truRelToAbsNumbering(truRelId);
  return isGood2x2(tileId) && try2x2(a, iTRU);
}

bool TriggerMap::isFiredMC4x4(float a, short iTRU, short ix, short iz) const
{
  char truRelId[3] = {char(iTRU), char(ix), char(iz)};
  short tileId = Geometry::truRelToAbsNumbering(truRelId);
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
