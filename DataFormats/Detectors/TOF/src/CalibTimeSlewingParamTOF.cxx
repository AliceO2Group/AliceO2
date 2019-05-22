// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibTimeSlewingParamTOF.cxx
/// \brief Class to store the output of the matching to TOF for calibration

#include <algorithm>
#include <cstdio>
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"

using namespace o2::dataformats;

//ClassImp(o2::dataformats::CalibTimeSlewingParamTOF);

CalibTimeSlewingParamTOF::CalibTimeSlewingParamTOF()
{

  mTimeSlewing[0] = &mTimeSlewingSec00;
  mTimeSlewing[1] = &mTimeSlewingSec01;
  mTimeSlewing[2] = &mTimeSlewingSec02;
  mTimeSlewing[3] = &mTimeSlewingSec03;
  mTimeSlewing[4] = &mTimeSlewingSec04;
  mTimeSlewing[5] = &mTimeSlewingSec05;
  mTimeSlewing[6] = &mTimeSlewingSec06;
  mTimeSlewing[7] = &mTimeSlewingSec07;
  mTimeSlewing[8] = &mTimeSlewingSec08;
  mTimeSlewing[9] = &mTimeSlewingSec09;
  mTimeSlewing[10] = &mTimeSlewingSec10;
  mTimeSlewing[11] = &mTimeSlewingSec11;
  mTimeSlewing[12] = &mTimeSlewingSec12;
  mTimeSlewing[13] = &mTimeSlewingSec13;
  mTimeSlewing[14] = &mTimeSlewingSec14;
  mTimeSlewing[15] = &mTimeSlewingSec15;
  mTimeSlewing[16] = &mTimeSlewingSec16;
  mTimeSlewing[17] = &mTimeSlewingSec17;

  for (int i = 0; i < NSECTORS; i++) {
    for (int j = 0; j < NCHANNELXSECTOR; j++) {
      mChannelStart[i][j] = -1;
      mFractionUnderPeak[i][j] = -100.;
      mSigmaPeak[i][j] = -1.;
    }
  }
}
//______________________________________________

float CalibTimeSlewingParamTOF::evalTimeSlewing(int channel, float tot) const
{
  int sector = channel / NCHANNELXSECTOR;
  channel = channel % NCHANNELXSECTOR;

  if (sector >= NSECTORS)
    return 0.; // something went wrong!

  int n = mChannelStart[sector][channel];
  if (n < 0)
    return 0.;

  int nstop = (*(mTimeSlewing[sector])).size();
  if (channel < NCHANNELXSECTOR - 1)
    nstop = mChannelStart[sector][channel + 1];

  if (n >= nstop)
    return 0.; // something went wrong!

  while (n < nstop && tot < (*(mTimeSlewing[sector]))[n].first)
    n++;
  n--;

  if (n < 0) { // tot is lower than the first available value
    return 0;
  }

  if (n == nstop - 1)
    return (*(mTimeSlewing[sector]))[n].second; // use the last value stored for that channel

  float w1 = tot - (*(mTimeSlewing[sector]))[n].first;
  float w2 = (*(mTimeSlewing[sector]))[n + 1].first - tot;

  return ((*(mTimeSlewing[sector]))[n].second * w2 + (*(mTimeSlewing[sector]))[n + 1].second * w1) / ((*(mTimeSlewing[sector]))[n + 1].first - (*(mTimeSlewing[sector]))[n].first);
}
//______________________________________________

void CalibTimeSlewingParamTOF::addTimeSlewingInfo(int channel, float tot, float time)
{
  // WE ARE ASSUMING THAT:
  // channels have to be filled in increasing order (within the sector)
  // tots have to be filled in increasing order (within the channel)
  int sector = channel / NCHANNELXSECTOR;
  channel = channel % NCHANNELXSECTOR;

  // printf("DBG: addTimeSlewinginfo sec=%i\n",sector);

  if (sector >= NSECTORS)
    return; // something went wrong!

  int currentch = channel;
  while (mChannelStart[sector][currentch] == -1 && currentch > -1) {
    // printf("DBG: fill channel %i\n",currentch);
    // set also all the previous ones which were not filled
    mChannelStart[sector][currentch] = (*(mTimeSlewing[sector])).size();
    currentch--;
  }
  // printf("DBG: emplace back (%f,%f)\n",tot,time);
  (*(mTimeSlewing[sector])).emplace_back(tot, time);
}
//______________________________________________

CalibTimeSlewingParamTOF& CalibTimeSlewingParamTOF::operator+=(const CalibTimeSlewingParamTOF& other)
{
  for (int i = 0; i < NSECTORS; i++) {
    if (other.mTimeSlewing[i]->size() > mTimeSlewing[i]->size()) {

      mTimeSlewing[i]->clear();
      for (auto obj = other.mTimeSlewing[i]->begin(); obj != other.mTimeSlewing[i]->end(); obj++)
        mTimeSlewing[i]->push_back(*obj);

      for (int j = 0; j < NCHANNELXSECTOR; j++) {
        mChannelStart[i][j] = other.mChannelStart[i][j];
        mFractionUnderPeak[i][j] = other.mFractionUnderPeak[i][j];
        mSigmaPeak[i][j] = other.mSigmaPeak[i][j];
      }
    }
  }
  return *this;
}
//______________________________________________
