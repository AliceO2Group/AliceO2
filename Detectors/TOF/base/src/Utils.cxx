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
int Utils::mNCalibTracks = 0;
o2::dataformats::CalibInfoTOF Utils::mCalibTracks[NTRACKS_REQUESTED];
int Utils::mNsample = 0;
int Utils::mIsample = 0;
float Utils::mPhases[100];

void Utils::addInteractionBC(int bc, bool fromCollisonCotext)
{
  if (fromCollisonCotext) { // align to TOF
    if (bc + Geo::LATENCY_ADJ_LHC_IN_BC < 0) {
      mFillScheme.push_back(bc + Geo::LATENCY_ADJ_LHC_IN_BC + Geo::BC_IN_ORBIT);
    } else if (bc + Geo::LATENCY_ADJ_LHC_IN_BC >= Geo::BC_IN_ORBIT) {
      mFillScheme.push_back(bc + Geo::LATENCY_ADJ_LHC_IN_BC - Geo::BC_IN_ORBIT);
    } else {
      mFillScheme.push_back(bc + Geo::LATENCY_ADJ_LHC_IN_BC);
    }
  } else {
    mFillScheme.push_back(bc);
  }
}

void Utils::init()
{
  memset(mBCmult, 0, o2::constants::lhc::LHCMaxBunches * sizeof(mBCmult[0]));
}

void Utils::addCalibTrack(float ctime)
{
  mCalibTracks[mNCalibTracks].setDeltaTimePi(ctime);

  mNCalibTracks++;

  if (mNCalibTracks >= NTRACKS_REQUESTED) {
    computeLHCphase();
    mNCalibTracks = 0;
  }
}

void Utils::computeLHCphase()
{
  static std::vector<o2::dataformats::CalibInfoTOF> tracks;
  tracks.clear();
  for (int i = 0; i < NTRACKS_REQUESTED; i++) {
    tracks.push_back(mCalibTracks[i]);
  }

  auto evtime = evTimeMaker<std::vector<o2::dataformats::CalibInfoTOF>, o2::dataformats::CalibInfoTOF, filterCalib<o2::dataformats::CalibInfoTOF>>(tracks, 6.0f, true);

  if (evtime.mEventTimeError < 100) { // udpate LHCphase
    mPhases[mIsample] = evtime.mEventTime;
    mIsample = (mIsample + 1) % 100;
    if (mNsample < 100) {
      mNsample++;
    }
  }

  mLHCPhase = 0;
  for (int i = 0; i < mNsample; i++) {
    mLHCPhase += mPhases[i];
  }
  mLHCPhase /= mNsample;
}

void Utils::printFillScheme()
{
  printf("FILL SCHEME\n");
  for (int i = 0; i < getNinteractionBC(); i++) {
    printf("BC(%d) LHCref=%d TOFref=%d\n", i, mFillScheme[i] - Geo::LATENCY_ADJ_LHC_IN_BC, mFillScheme[i]);
  }
}

int Utils::getNinteractionBC()
{
  return mFillScheme.size();
}

double Utils::subtractInteractionBC(double time, int& mask, bool subLatency)
{
  static const int deltalat = o2::tof::Geo::BC_IN_ORBIT - o2::tof::Geo::LATENCYWINDOW_IN_BC;
  int bc = int(time * o2::tof::Geo::BC_TIME_INPS_INV + 0.2);

  if (subLatency) {
    if (bc >= o2::tof::Geo::LATENCYWINDOW_IN_BC) {
      bc -= o2::tof::Geo::LATENCYWINDOW_IN_BC;
      time -= o2::tof::Geo::LATENCYWINDOW_IN_BC * o2::tof::Geo::BC_TIME_INPS;
    } else {
      bc += deltalat;
      time += deltalat * o2::tof::Geo::BC_TIME_INPS;
    }
  }

  int bcOrbit = bc % o2::constants::lhc::LHCMaxBunches;

  int dbc = o2::constants::lhc::LHCMaxBunches, bcc = bc;
  int dbcSigned = 1000;
  for (int k = 0; k < getNinteractionBC(); k++) { // get bc from fill scheme closest
    int deltaCBC = bcOrbit - getInteractionBC(k);
    if (deltaCBC >= -8 && deltaCBC < 8) {
      mask += (1 << (deltaCBC + 8)); // fill bc candidates
    }
    if (abs(deltaCBC) < dbc) {
      bcc = bc - deltaCBC;
      dbcSigned = deltaCBC;
      dbc = abs(dbcSigned);
    }
    if (abs(deltaCBC + o2::constants::lhc::LHCMaxBunches) < dbc) { // in case k is close to the right border (last BC of the orbit)
      bcc = bc - deltaCBC - o2::constants::lhc::LHCMaxBunches;
      dbcSigned = deltaCBC + o2::constants::lhc::LHCMaxBunches;
      dbc = abs(dbcSigned);
    }
    if (abs(deltaCBC - o2::constants::lhc::LHCMaxBunches) < dbc) { // in case k is close to the left border (BC=0)
      bcc = bc - deltaCBC + o2::constants::lhc::LHCMaxBunches;
      dbcSigned = deltaCBC - o2::constants::lhc::LHCMaxBunches;
      dbc = abs(dbcSigned);
    }
  }
  if (dbcSigned >= -8 && dbcSigned < 8) {
    mask += (1 << (dbcSigned + 24)); // fill bc used
  }

  time -= o2::tof::Geo::BC_TIME_INPS * bcc;

  return time;
}

float Utils::subtractInteractionBC(float time, int& mask, bool subLatency)
{
  static const int deltalat = o2::tof::Geo::BC_IN_ORBIT - o2::tof::Geo::LATENCYWINDOW_IN_BC;
  int bc = int(time * o2::tof::Geo::BC_TIME_INPS_INV + 0.2);

  if (subLatency) {
    if (bc >= o2::tof::Geo::LATENCYWINDOW_IN_BC) {
      bc -= o2::tof::Geo::LATENCYWINDOW_IN_BC;
      time -= o2::tof::Geo::LATENCYWINDOW_IN_BC * o2::tof::Geo::BC_TIME_INPS;
    } else {
      bc += deltalat;
      time += deltalat * o2::tof::Geo::BC_TIME_INPS;
    }
  }

  int bcOrbit = bc % o2::constants::lhc::LHCMaxBunches;

  int dbc = o2::constants::lhc::LHCMaxBunches, bcc = bc;
  int dbcSigned = 1000;
  for (int k = 0; k < getNinteractionBC(); k++) { // get bc from fill scheme closest
    int deltaCBC = bcOrbit - getInteractionBC(k);
    if (deltaCBC >= -8 && deltaCBC < 8) {
      mask += (1 << (deltaCBC + 8)); // fill bc candidates
    }
    if (abs(deltaCBC) < dbc) {
      bcc = bc - deltaCBC;
      dbcSigned = deltaCBC;
      dbc = abs(dbcSigned);
    }
    if (abs(deltaCBC + o2::constants::lhc::LHCMaxBunches) < dbc) { // in case k is close to the right border (last BC of the orbit)
      bcc = bc - deltaCBC - o2::constants::lhc::LHCMaxBunches;
      dbcSigned = deltaCBC + o2::constants::lhc::LHCMaxBunches;
      dbc = abs(dbcSigned);
    }
    if (abs(deltaCBC - o2::constants::lhc::LHCMaxBunches) < dbc) { // in case k is close to the left border (BC=0)
      bcc = bc - deltaCBC + o2::constants::lhc::LHCMaxBunches;
      dbcSigned = deltaCBC - o2::constants::lhc::LHCMaxBunches;
      dbc = abs(dbcSigned);
    }
  }
  if (dbcSigned >= -8 && dbcSigned < 8) {
    mask += (1 << (dbcSigned + 24)); // fill bc used
  }

  time -= o2::tof::Geo::BC_TIME_INPS * bcc;

  return time;
}

void Utils::addBC(float toftime, bool subLatency)
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
  static const int deltalat = o2::tof::Geo::BC_IN_ORBIT - o2::tof::Geo::LATENCYWINDOW_IN_BC;
  int bc = int(toftime * o2::tof::Geo::BC_TIME_INPS_INV + 0.2) % o2::constants::lhc::LHCMaxBunches;

  if (subLatency) {
    if (bc >= o2::tof::Geo::LATENCYWINDOW_IN_BC) {
      bc -= o2::tof::Geo::LATENCYWINDOW_IN_BC;
      toftime -= o2::tof::Geo::LATENCYWINDOW_IN_BC * o2::tof::Geo::BC_TIME_INPS;
    } else {
      bc += deltalat;
      toftime += deltalat * o2::tof::Geo::BC_TIME_INPS;
    }
  }

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
