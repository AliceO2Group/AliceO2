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
  for (int i = 0; i < NSECTORS; i++) {
    memset(mChannelStart[i], -1, sizeof(mChannelStart[i]));
    memset(mFractionUnderPeak[i], -100, sizeof(mFractionUnderPeak[i]));
    memset(mSigmaPeak[i], -1., sizeof(mSigmaPeak[i]));
  }
}
//______________________________________________
CalibTimeSlewingParamTOF::CalibTimeSlewingParamTOF(const CalibTimeSlewingParamTOF& source)
{

  for (int iSec = 0; iSec < NSECTORS; iSec++) {
    mTimeSlewing[iSec] = source.mTimeSlewing[iSec];
  }
  for (int i = 0; i < NSECTORS; i++) {
    memcpy(mChannelStart[i], source.mChannelStart[i], sizeof(mChannelStart[i]));
    memcpy(mFractionUnderPeak[i], source.mFractionUnderPeak[i], sizeof(mFractionUnderPeak[i]));
    memcpy(mSigmaPeak[i], source.mSigmaPeak[i], sizeof(mSigmaPeak[i]));
  }
}
//______________________________________________
CalibTimeSlewingParamTOF& CalibTimeSlewingParamTOF::operator=(const CalibTimeSlewingParamTOF& source)
{
  for (int i = 0; i < NSECTORS; i++) {
    mTimeSlewing[i].clear(); // CHECK: mTimeSlewing[i] does not work, probably because it is an array of vectors
    mTimeSlewing[i] = source.mTimeSlewing[i];
    memcpy(mChannelStart[i], source.mChannelStart[i], sizeof(mChannelStart[i]));
    memcpy(mFractionUnderPeak[i], source.mFractionUnderPeak[i], sizeof(mFractionUnderPeak[i]));
    memcpy(mSigmaPeak[i], source.mSigmaPeak[i], sizeof(mSigmaPeak[i]));
  }
  return *this;
}

//______________________________________________
float CalibTimeSlewingParamTOF::getChannelOffset(int channel) const
{
  return evalTimeSlewing(channel, 0);
}

//______________________________________________
float CalibTimeSlewingParamTOF::evalTimeSlewing(int channel, float totIn) const
{

  // totIn is in ns
  // the correction is returned in ps

  int sector = channel / NCHANNELXSECTOR;
  channel = channel % NCHANNELXSECTOR;

  if (sector >= NSECTORS)
    return 0.; // something went wrong!

  int n = mChannelStart[sector][channel];
  if (n < 0)
    return 0.;

  int nstop = (mTimeSlewing[sector]).size();
  if (channel < NCHANNELXSECTOR - 1)
    nstop = mChannelStart[sector][channel + 1];

  if (n >= nstop)
    return 0.; // something went wrong!

  if (totIn == 0) {
    return (float)((mTimeSlewing[sector])[n].second);
  }

  // we convert tot from ns to ps and to unsigned short
  unsigned short tot = (unsigned short)(totIn * 1000);

  while (n < nstop && tot > (mTimeSlewing[sector])[n].first)
    n++;
  n--;

  if (n < 0) { // tot is lower than the first available value
    return 0;
  }

  if (n == nstop - 1)
    return (float)((mTimeSlewing[sector])[n].second); // use the last value stored for that channel

  float w1 = (float)(tot - (mTimeSlewing[sector])[n].first);
  float w2 = (float)((mTimeSlewing[sector])[n + 1].first - tot);

  return (float)(((mTimeSlewing[sector])[n].second * w2 + (mTimeSlewing[sector])[n + 1].second * w1) / ((mTimeSlewing[sector])[n + 1].first - (mTimeSlewing[sector])[n].first));
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

  if (sector >= NSECTORS)
    return; // something went wrong!

  int currentch = channel;
  while (mChannelStart[sector][currentch] == -1 && currentch > -1) {
    // printf("DBG: fill channel %i\n",currentch);
    // set also all the previous ones which were not filled
    mChannelStart[sector][currentch] = (mTimeSlewing[sector]).size();
    currentch--;
  }
  // printf("DBG: emplace back (%f,%f)\n",tot,time);
  (mTimeSlewing[sector]).emplace_back((unsigned short)(tot * 1000), (short)time);
}
//______________________________________________

bool CalibTimeSlewingParamTOF::updateOffsetInfo(int channel, float residualOffset)
{

  // to update only the channel offset info in an existing CCDB object
  // residual offset is given in ps

  int sector = channel / NCHANNELXSECTOR;
  channel = channel % NCHANNELXSECTOR;
  //  printf("sector = %d, channel = %d\n", sector, channel);

  // printf("DBG: addTimeSlewinginfo sec=%i\n",sector);

  int n = mChannelStart[sector][channel]; // first time slewing entry for the current channel. this corresponds to tot = 0
  if ((mTimeSlewing[sector])[n].first != 0) {
    printf("DBG: there was no time offset set yet! first tot is %d\n", (mTimeSlewing[sector])[n].first);
    std::pair<unsigned short, short> offsetToBeInserted(0, (short)residualOffset);
    auto it = (mTimeSlewing[sector]).begin();
    (mTimeSlewing[sector]).insert(it + n, offsetToBeInserted);
    // now we have to increase by 1 all the mChannelStart for the channels that come after this
    for (auto ch = channel + 1; ch < NCHANNELXSECTOR; ch++) {
      mChannelStart[sector][ch]++;
    }
    return false;
  }
  (mTimeSlewing[sector])[n].second += (short)residualOffset;
  return true;
}
//______________________________________________

CalibTimeSlewingParamTOF& CalibTimeSlewingParamTOF::operator+=(const CalibTimeSlewingParamTOF& other)
{
  for (int i = 0; i < NSECTORS; i++) {
    if (other.mTimeSlewing[i].size() > mTimeSlewing[i].size()) {

      mTimeSlewing[i].clear();
      for (auto obj = other.mTimeSlewing[i].begin(); obj != other.mTimeSlewing[i].end(); obj++)
        mTimeSlewing[i].push_back(*obj);

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
