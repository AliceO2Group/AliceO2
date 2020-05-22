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
CalibTimeSlewingParamTOF::CalibTimeSlewingParamTOF(const CalibTimeSlewingParamTOF& source)
{
  mTimeSlewingSec00 = source.mTimeSlewingSec00;
  mTimeSlewingSec01 = source.mTimeSlewingSec01;
  mTimeSlewingSec02 = source.mTimeSlewingSec02;
  mTimeSlewingSec03 = source.mTimeSlewingSec03;
  mTimeSlewingSec04 = source.mTimeSlewingSec04;
  mTimeSlewingSec05 = source.mTimeSlewingSec05;
  mTimeSlewingSec06 = source.mTimeSlewingSec06;
  mTimeSlewingSec07 = source.mTimeSlewingSec07;
  mTimeSlewingSec08 = source.mTimeSlewingSec08;
  mTimeSlewingSec09 = source.mTimeSlewingSec09;
  mTimeSlewingSec10 = source.mTimeSlewingSec10;
  mTimeSlewingSec11 = source.mTimeSlewingSec11;
  mTimeSlewingSec12 = source.mTimeSlewingSec12;
  mTimeSlewingSec13 = source.mTimeSlewingSec13;
  mTimeSlewingSec14 = source.mTimeSlewingSec14;
  mTimeSlewingSec15 = source.mTimeSlewingSec15;
  mTimeSlewingSec16 = source.mTimeSlewingSec16;
  mTimeSlewingSec17 = source.mTimeSlewingSec17;
  /*
  for (int i = 0; i < source.mTimeSlewingSec00.size(); i++) {
    //printf("i = %d, value first = %f, value second = %f\n", i, source.mTimeSlewingSec00[i].first, source.mTimeSlewingSec00[i].second);
    mTimeSlewingSec00.push_back(source.mTimeSlewingSec00[i]);
  }
  
  for (int i = 0; i < source.mTimeSlewingSec01.size(); i++) mTimeSlewingSec01.push_back(source.mTimeSlewingSec01[i]);
  for (int i = 0; i < source.mTimeSlewingSec02.size(); i++) mTimeSlewingSec02.push_back(source.mTimeSlewingSec02[i]);
  for (int i = 0; i < source.mTimeSlewingSec03.size(); i++) mTimeSlewingSec03.push_back(source.mTimeSlewingSec03[i]);
  for (int i = 0; i < source.mTimeSlewingSec04.size(); i++) mTimeSlewingSec04.push_back(source.mTimeSlewingSec04[i]);
  for (int i = 0; i < source.mTimeSlewingSec05.size(); i++) mTimeSlewingSec05.push_back(source.mTimeSlewingSec05[i]);
  for (int i = 0; i < source.mTimeSlewingSec06.size(); i++) mTimeSlewingSec06.push_back(source.mTimeSlewingSec06[i]);
  for (int i = 0; i < source.mTimeSlewingSec07.size(); i++) mTimeSlewingSec07.push_back(source.mTimeSlewingSec07[i]);
  for (int i = 0; i < source.mTimeSlewingSec08.size(); i++) mTimeSlewingSec08.push_back(source.mTimeSlewingSec08[i]);
  for (int i = 0; i < source.mTimeSlewingSec09.size(); i++) mTimeSlewingSec09.push_back(source.mTimeSlewingSec09[i]);
  for (int i = 0; i < source.mTimeSlewingSec10.size(); i++) mTimeSlewingSec10.push_back(source.mTimeSlewingSec10[i]);
  for (int i = 0; i < source.mTimeSlewingSec11.size(); i++) mTimeSlewingSec11.push_back(source.mTimeSlewingSec11[i]);
  for (int i = 0; i < source.mTimeSlewingSec12.size(); i++) mTimeSlewingSec12.push_back(source.mTimeSlewingSec12[i]);
  for (int i = 0; i < source.mTimeSlewingSec13.size(); i++) mTimeSlewingSec13.push_back(source.mTimeSlewingSec13[i]);
  for (int i = 0; i < source.mTimeSlewingSec14.size(); i++) mTimeSlewingSec14.push_back(source.mTimeSlewingSec14[i]);
  for (int i = 0; i < source.mTimeSlewingSec15.size(); i++) mTimeSlewingSec15.push_back(source.mTimeSlewingSec15[i]);
  for (int i = 0; i < source.mTimeSlewingSec16.size(); i++) mTimeSlewingSec16.push_back(source.mTimeSlewingSec16[i]);
  for (int i = 0; i < source.mTimeSlewingSec17.size(); i++) mTimeSlewingSec17.push_back(source.mTimeSlewingSec17[i]);
  */
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
      mChannelStart[i][j] = source.mChannelStart[i][j];
      mFractionUnderPeak[i][j] = source.mFractionUnderPeak[i][j];
      mSigmaPeak[i][j] = source.mSigmaPeak[i][j];
    }
  }
}
//______________________________________________
CalibTimeSlewingParamTOF& CalibTimeSlewingParamTOF::operator=(const CalibTimeSlewingParamTOF& source)
{
  for (int i = 0; i < NSECTORS; i++) {
    mTimeSlewing[i]->clear();
    *(mTimeSlewing[i]) = *(source.mTimeSlewing[i]);

    for (int j = 0; j < NCHANNELXSECTOR; j++) {
      mChannelStart[i][j] = source.mChannelStart[i][j];
      mFractionUnderPeak[i][j] = source.mFractionUnderPeak[i][j];
      mSigmaPeak[i][j] = source.mSigmaPeak[i][j];
    }
  }
  return *this;
}

//______________________________________________
float CalibTimeSlewingParamTOF::getChannelOffset(int channel) const
{
  return evalTimeSlewing(channel, 0);
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

  if (tot == 0) {
    return (*(mTimeSlewing[sector]))[n].second;
  }

  int nstop = (*(mTimeSlewing[sector])).size();
  if (channel < NCHANNELXSECTOR - 1)
    nstop = mChannelStart[sector][channel + 1];

  if (n >= nstop)
    return 0.; // something went wrong!

  while (n < nstop && tot > (*(mTimeSlewing[sector]))[n].first)
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

bool CalibTimeSlewingParamTOF::updateOffsetInfo(int channel, float residualOffset)
{
  
  // to update only the channel offset info in an existing CCDB object

  int sector = channel / NCHANNELXSECTOR;
  channel = channel % NCHANNELXSECTOR;
  //  printf("sector = %d, channel = %d\n", sector, channel);

  // printf("DBG: addTimeSlewinginfo sec=%i\n",sector);

  int n = mChannelStart[sector][channel]; // first time slewing entry for the current channel. this corresponds to tot = 0
  if ((*(mTimeSlewing[sector]))[n].first != 0) {
    printf("DBG: there was no time offset set yet! first tot is %f\n", (*(mTimeSlewing[sector]))[n].first);
    std::pair<float, float> offsetToBeInserted(0, residualOffset);
    auto it = (*(mTimeSlewing[sector])).begin();
    (*(mTimeSlewing[sector])).insert(it+n, offsetToBeInserted);
    // now we have to increase by 1 all the mChannelStart for the channels that come after this
    for (auto ch = channel+1; ch < NCHANNELXSECTOR; ch++){
      mChannelStart[sector][ch]++;
    }
    return false;
  }
  (*(mTimeSlewing[sector]))[n].second += residualOffset;
  return true;
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
