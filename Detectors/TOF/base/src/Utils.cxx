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

//
//  Strip.cxx: structure to store the TOF digits in strips - useful
// for clusterization purposes
//  ALICEO2
//
#include <TMath.h>

#include "TOFBase/Utils.h"

#define MAX_NUM_EVENT_AUTODETECT 10000

using namespace o2::tof;

ClassImp(o2::tof::Utils);

std::vector<int> Utils::mFillScheme;
int Utils::mBCmult[o2::constants::lhc::LHCMaxBunches];
int Utils::mNautodet = 0;
int Utils::mMaxBC = 0;
bool Utils::mIsInit = false;
float Utils::mEventTimeSpread = 200;
float Utils::mEtaMin = -0.8;
float Utils::mEtaMax = 0.8;
float Utils::mLHCPhase = 0;

void Utils::init()
{
  memset(mBCmult, 0, o2::constants::lhc::LHCMaxBunches * sizeof(mBCmult[0]));
}

void Utils::printFillScheme()
{
  printf("FILL SCHEME\n");
  for (int i = 0; i < getNinteractionBC(); i++) {
    printf("BC(%d) %d\n", i, mFillScheme[i]);
  }
}

int Utils::getNinteractionBC()
{
  return mFillScheme.size();
}

double Utils::subtractInteractionBC(double time)
{
  int bc = int(time * o2::tof::Geo::BC_TIME_INPS_INV + 0.2);
  int bcOrbit = bc % o2::constants::lhc::LHCMaxBunches;

  int dbc = o2::constants::lhc::LHCMaxBunches, bcc = bc;
  for (int k = 0; k < getNinteractionBC(); k++) { // get bc from fill scheme closest
    if (abs(bcOrbit - getInteractionBC(k)) < dbc) {
      bcc = bc - bcOrbit + getInteractionBC(k);
      dbc = abs(bcOrbit - getInteractionBC(k));
    }
    if (abs(bcOrbit - getInteractionBC(k) + o2::constants::lhc::LHCMaxBunches) < dbc) { // in case k is close to the right border (last BC of the orbit)
      bcc = bc - bcOrbit + getInteractionBC(k) - o2::constants::lhc::LHCMaxBunches;
      dbc = abs(bcOrbit - getInteractionBC(k) + o2::constants::lhc::LHCMaxBunches);
    }
    if (abs(bcOrbit - getInteractionBC(k) - o2::constants::lhc::LHCMaxBunches) < dbc) { // in case k is close to the left border (BC=0)
      bcc = bc - bcOrbit + getInteractionBC(k) + o2::constants::lhc::LHCMaxBunches;
      dbc = abs(bcOrbit - getInteractionBC(k) - o2::constants::lhc::LHCMaxBunches);
    }
  }
  time -= o2::tof::Geo::BC_TIME_INPS * bcc;

  return time;
}

float Utils::subtractInteractionBC(float time)
{
  int bc = int(time * o2::tof::Geo::BC_TIME_INPS_INV + 0.2);
  int bcOrbit = bc % o2::constants::lhc::LHCMaxBunches;

  int dbc = o2::constants::lhc::LHCMaxBunches, bcc = bc;
  for (int k = 0; k < getNinteractionBC(); k++) { // get bc from fill scheme closest
    if (abs(bcOrbit - getInteractionBC(k)) < dbc) {
      bcc = bc - bcOrbit + getInteractionBC(k);
      dbc = abs(bcOrbit - getInteractionBC(k));
    }
  }
  time -= o2::tof::Geo::BC_TIME_INPS * bcc;

  return time;
}

void Utils::addBC(float toftime)
{
  if (!mIsInit) {
    init();
    mIsInit = true;
  }

  if (mNautodet > MAX_NUM_EVENT_AUTODETECT) {
    if (!hasFillScheme()) { // detect fill scheme
      int thres = mMaxBC / 2;
      for (int i = 0; i < o2::constants::lhc::LHCMaxBunches; i++) {
        if (mBCmult[i] > thres) { // good bunch
          addInteractionBC(i);
        }
      }
    }
    return;
  }

  // just fill
  int bc = int(toftime * o2::tof::Geo::BC_TIME_INPS_INV + 0.2) % o2::constants::lhc::LHCMaxBunches;

  mBCmult[bc]++;

  if (mBCmult[bc] > mMaxBC) {
    mMaxBC = mBCmult[bc];
  }
  mNautodet++;
}

bool Utils::hasFillScheme()
{
  if (getNinteractionBC()) {
    return true;
  }

  return false;
}
