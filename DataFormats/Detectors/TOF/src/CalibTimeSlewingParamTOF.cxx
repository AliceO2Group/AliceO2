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

/// \file CalibTimeSlewingParamTOF.cxx
/// \brief Class to store the output of the matching to TOF for calibration

#include <algorithm>
#include <cstdio>
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"

using namespace o2::dataformats;

CalibTimeSlewingParamTOF::CalibTimeSlewingParamTOF()
{

  for (int i = 0; i < NSECTORS; i++) {
    memset((*(mChannelStart[i])).data(), -1, sizeof(*(mChannelStart[i])));
    for (int j = 0; j < NCHANNELXSECTOR; j++) {
      (*(mFractionUnderPeak[i]))[j] = -100.;
      (*(mSigmaPeak[i]))[j] = -1.;
    }
  }
}

//______________________________________________
float CalibTimeSlewingParamTOF::getChannelOffset(int channel) const
{
  return evalTimeSlewing(channel, 0);
}

//______________________________________________
void CalibTimeSlewingParamTOF::setChannelOffset(int channel, float val)
{
  int sector = channel / NCHANNELXSECTOR;
  channel = channel % NCHANNELXSECTOR;

  (*(mGlobalOffset[sector]))[channel] = val;
}
//______________________________________________
void CalibTimeSlewingParamTOF::bind()
{
  mGlobalOffset[0] = &mGlobalOffsetSec0;
  mGlobalOffset[1] = &mGlobalOffsetSec1;
  mGlobalOffset[2] = &mGlobalOffsetSec2;
  mGlobalOffset[3] = &mGlobalOffsetSec3;
  mGlobalOffset[4] = &mGlobalOffsetSec4;
  mGlobalOffset[5] = &mGlobalOffsetSec5;
  mGlobalOffset[6] = &mGlobalOffsetSec6;
  mGlobalOffset[7] = &mGlobalOffsetSec7;
  mGlobalOffset[8] = &mGlobalOffsetSec8;
  mGlobalOffset[9] = &mGlobalOffsetSec9;
  mGlobalOffset[10] = &mGlobalOffsetSec10;
  mGlobalOffset[11] = &mGlobalOffsetSec11;
  mGlobalOffset[12] = &mGlobalOffsetSec12;
  mGlobalOffset[13] = &mGlobalOffsetSec13;
  mGlobalOffset[14] = &mGlobalOffsetSec14;
  mGlobalOffset[15] = &mGlobalOffsetSec15;
  mGlobalOffset[16] = &mGlobalOffsetSec16;
  mGlobalOffset[17] = &mGlobalOffsetSec17;

  mChannelStart[0] = &mChannelStartSec0;
  mChannelStart[1] = &mChannelStartSec1;
  mChannelStart[2] = &mChannelStartSec2;
  mChannelStart[3] = &mChannelStartSec3;
  mChannelStart[4] = &mChannelStartSec4;
  mChannelStart[5] = &mChannelStartSec5;
  mChannelStart[6] = &mChannelStartSec6;
  mChannelStart[7] = &mChannelStartSec7;
  mChannelStart[8] = &mChannelStartSec8;
  mChannelStart[9] = &mChannelStartSec9;
  mChannelStart[10] = &mChannelStartSec10;
  mChannelStart[11] = &mChannelStartSec11;
  mChannelStart[12] = &mChannelStartSec12;
  mChannelStart[13] = &mChannelStartSec13;
  mChannelStart[14] = &mChannelStartSec14;
  mChannelStart[15] = &mChannelStartSec15;
  mChannelStart[16] = &mChannelStartSec16;
  mChannelStart[17] = &mChannelStartSec17;

  mTimeSlewing[0] = &mTimeSlewingSec0;
  mTimeSlewing[1] = &mTimeSlewingSec1;
  mTimeSlewing[2] = &mTimeSlewingSec2;
  mTimeSlewing[3] = &mTimeSlewingSec3;
  mTimeSlewing[4] = &mTimeSlewingSec4;
  mTimeSlewing[5] = &mTimeSlewingSec5;
  mTimeSlewing[6] = &mTimeSlewingSec6;
  mTimeSlewing[7] = &mTimeSlewingSec7;
  mTimeSlewing[8] = &mTimeSlewingSec8;
  mTimeSlewing[9] = &mTimeSlewingSec9;
  mTimeSlewing[10] = &mTimeSlewingSec10;
  mTimeSlewing[11] = &mTimeSlewingSec11;
  mTimeSlewing[12] = &mTimeSlewingSec12;
  mTimeSlewing[13] = &mTimeSlewingSec13;
  mTimeSlewing[14] = &mTimeSlewingSec14;
  mTimeSlewing[15] = &mTimeSlewingSec15;
  mTimeSlewing[16] = &mTimeSlewingSec16;
  mTimeSlewing[17] = &mTimeSlewingSec17;

  mFractionUnderPeak[0] = &mFractionUnderPeakSec0;
  mFractionUnderPeak[1] = &mFractionUnderPeakSec1;
  mFractionUnderPeak[2] = &mFractionUnderPeakSec2;
  mFractionUnderPeak[3] = &mFractionUnderPeakSec3;
  mFractionUnderPeak[4] = &mFractionUnderPeakSec4;
  mFractionUnderPeak[5] = &mFractionUnderPeakSec5;
  mFractionUnderPeak[6] = &mFractionUnderPeakSec6;
  mFractionUnderPeak[7] = &mFractionUnderPeakSec7;
  mFractionUnderPeak[8] = &mFractionUnderPeakSec8;
  mFractionUnderPeak[9] = &mFractionUnderPeakSec9;
  mFractionUnderPeak[10] = &mFractionUnderPeakSec10;
  mFractionUnderPeak[11] = &mFractionUnderPeakSec11;
  mFractionUnderPeak[12] = &mFractionUnderPeakSec12;
  mFractionUnderPeak[13] = &mFractionUnderPeakSec13;
  mFractionUnderPeak[14] = &mFractionUnderPeakSec14;
  mFractionUnderPeak[15] = &mFractionUnderPeakSec15;
  mFractionUnderPeak[16] = &mFractionUnderPeakSec16;
  mFractionUnderPeak[17] = &mFractionUnderPeakSec17;

  mSigmaPeak[0] = &mSigmaPeakSec0;
  mSigmaPeak[1] = &mSigmaPeakSec1;
  mSigmaPeak[2] = &mSigmaPeakSec2;
  mSigmaPeak[3] = &mSigmaPeakSec3;
  mSigmaPeak[4] = &mSigmaPeakSec4;
  mSigmaPeak[5] = &mSigmaPeakSec5;
  mSigmaPeak[6] = &mSigmaPeakSec6;
  mSigmaPeak[7] = &mSigmaPeakSec7;
  mSigmaPeak[8] = &mSigmaPeakSec8;
  mSigmaPeak[9] = &mSigmaPeakSec9;
  mSigmaPeak[10] = &mSigmaPeakSec10;
  mSigmaPeak[11] = &mSigmaPeakSec11;
  mSigmaPeak[12] = &mSigmaPeakSec12;
  mSigmaPeak[13] = &mSigmaPeakSec13;
  mSigmaPeak[14] = &mSigmaPeakSec14;
  mSigmaPeak[15] = &mSigmaPeakSec15;
  mSigmaPeak[16] = &mSigmaPeakSec16;
  mSigmaPeak[17] = &mSigmaPeakSec17;
}

//______________________________________________
float CalibTimeSlewingParamTOF::evalTimeSlewing(int channel, float totIn) const
{
  // totIn is in ns
  // the correction is returned in ps

  int sector = channel / NCHANNELXSECTOR;
  channel = channel % NCHANNELXSECTOR;

  if (sector >= NSECTORS) {
    return 0.; // something went wrong!
  }

  int n = (*(mChannelStart[sector]))[channel];
  if (n < 0) {
    return 0.;
  }
  int nstop = mTimeSlewing[sector]->size();

  if (channel < NCHANNELXSECTOR - 1) {
    nstop = (*(mChannelStart[sector]))[channel + 1];
  }

  if (n >= nstop) {
    return 0.; // something went wrong!
  }

  if (totIn == 0) {
    return (float)((*(mTimeSlewing[sector]))[n].second + (*(mGlobalOffset[sector]))[channel]);
  }

  // we convert tot from ns to ps and to unsigned short
  unsigned short tot = (unsigned short)(totIn * 1000);

  while (n < nstop && tot > (*(mTimeSlewing[sector]))[n].first) {
    n++;
  }
  n--;

  if (n < 0) { // tot is lower than the first available value
    return 0;
  }

  if (n == nstop - 1) {
    return (float)((*(mTimeSlewing[sector]))[n].second + (*(mGlobalOffset[sector]))[channel]); // use the last value stored for that channel
  }

  float w1 = (float)(tot - (*(mTimeSlewing[sector]))[n].first);
  float w2 = (float)((*(mTimeSlewing[sector]))[n + 1].first - tot);

  return (float)((*(mGlobalOffset[sector]))[channel] + (((*(mTimeSlewing[sector]))[n].second * w2 + (*(mTimeSlewing[sector]))[n + 1].second * w1) / ((*(mTimeSlewing[sector]))[n + 1].first - (*(mTimeSlewing[sector]))[n].first)));
}
//______________________________________________

void CalibTimeSlewingParamTOF::addTimeSlewingInfo(int channel, float tot, float time)
{
  // WE ARE ASSUMING THAT:
  // channels have to be filled in increasing order (within the sector)
  // tots have to be filled in increasing order (within the channel)

  // tot here is provided in ns, time in ps;
  // tot will have to be converted into ps;
  // type will have to be converted to unsigned short (tot) and short (time)

  int sector = channel / NCHANNELXSECTOR;
  channel = channel % NCHANNELXSECTOR;

  // printf("DBG: addTimeSlewinginfo sec=%i\n",sector);

  if (sector >= NSECTORS) {
    return; // something went wrong!
  }

  int currentch = channel;
  while (currentch > -1 && (*(mChannelStart[sector]))[currentch] == -1) {
    // printf("DBG: fill channel %i\n",currentch);
    // set also all the previous ones which were not filled
    (*(mChannelStart[sector]))[currentch] = mTimeSlewing[sector]->size();
    (*(mGlobalOffset[sector]))[currentch] = time;
    currentch--;
  }
  // printf("DBG: emplace back (%f,%f)\n",tot,time);
  (*(mTimeSlewing[sector])).emplace_back((unsigned short)(tot * 1000), (short)(time - (*(mGlobalOffset[sector]))[channel]));
}
//______________________________________________

bool CalibTimeSlewingParamTOF::updateOffsetInfo(int channel, float residualOffset)
{

  // to update only the channel offset info in an existing CCDB object
  // residual offset is given in ps

  int sector = channel / NCHANNELXSECTOR;
  channel = channel % NCHANNELXSECTOR;
  //  printf("sector = %d, channel = %d\n", sector, channel);

  (*(mGlobalOffset[sector]))[channel] += residualOffset;
  return true;

  /*
  // printf("DBG: addTimeSlewinginfo sec=%i\n",sector);

  int n = (*(mChannelStart[sector]))[channel]; // first time slewing entry for the current channel. this corresponds to tot = 0
  if ((*(mTimeSlewing[sector]))[n].first != 0) {
    printf("DBG: there was no time offset set yet! first tot is %d\n", (*(mTimeSlewing[sector]))[n].first);
    std::pair<unsigned short, short> offsetToBeInserted(0, (short)residualOffset);
    auto it = (*(mTimeSlewing[sector])).begin();
    (*(mTimeSlewing[sector])).insert(it + n, offsetToBeInserted);
    // now we have to increase by 1 all the mChannelStart for the channels that come after this
    for (auto ch = channel + 1; ch < NCHANNELXSECTOR; ch++) {
      (*(mChannelStart[sector]))[ch]++;
    }
    return false;
  }
  (*(mTimeSlewing[sector]))[n].second += (short)residualOffset;
  return true;
*/
}
//______________________________________________
CalibTimeSlewingParamTOF& CalibTimeSlewingParamTOF::operator+=(const CalibTimeSlewingParamTOF& other)
{
  for (int i = 0; i < NSECTORS; i++) {
    if (other.mTimeSlewing[i]->size() > mTimeSlewing[i]->size()) {
      *(mTimeSlewing[i]) = *(other.mTimeSlewing[i]);
      (*(mChannelStart[i])) = (*(other.mChannelStart[i]));
      *(mFractionUnderPeak[i]) = *(other.mFractionUnderPeak[i]);
      *(mSigmaPeak[i]) = *(other.mSigmaPeak[i]);
      *(mGlobalOffset[i]) = *(other.mGlobalOffset[i]);
    }
  }
  return *this;
}
//______________________________________________
CalibTimeSlewingParamTOF::CalibTimeSlewingParamTOF(const CalibTimeSlewingParamTOF& source)
{
  bind();
  for (int i = 0; i < NSECTORS; i++) {
    *(mTimeSlewing[i]) = *(source.mTimeSlewing[i]);
    (*(mChannelStart[i])) = (*(source.mChannelStart[i]));
    *(mFractionUnderPeak[i]) = *(source.mFractionUnderPeak[i]);
    *(mSigmaPeak[i]) = *(source.mSigmaPeak[i]);
    *(mGlobalOffset[i]) = *(source.mGlobalOffset[i]);
  }
  mStartValidity = source.mStartValidity;
  mEndValidity = source.mEndValidity;
}
