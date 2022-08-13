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

/// \file CalibTimeSlewingParamTOF.h
/// \brief Class to store the output of the matching to TOF for calibration

#ifndef ALICEO2_CALIBTIMESLEWINGPARAMTOF_H
#define ALICEO2_CALIBTIMESLEWINGPARAMTOF_H

#include <vector>
#include <array>
#include "Rtypes.h"

namespace o2
{
namespace dataformats
{
class CalibTimeSlewingParamTOF
{
 public:
  static const int NCHANNELS = 157248;                     //
  static const int NSECTORS = 18;                          //
  static const int NCHANNELXSECTOR = NCHANNELS / NSECTORS; //

  CalibTimeSlewingParamTOF();

  CalibTimeSlewingParamTOF(const CalibTimeSlewingParamTOF& source);

  CalibTimeSlewingParamTOF& operator=(const CalibTimeSlewingParamTOF& source) = default;

  float getChannelOffset(int channel) const;
  void setChannelOffset(int channel, float val);

  float evalTimeSlewing(int channel, float tot) const;

  void addTimeSlewingInfo(int channel, float tot, float time);

  bool updateOffsetInfo(int channel, float residualOffset);

  const std::vector<std::pair<unsigned short, short>>& getVector(int sector) const { return *(mTimeSlewing[sector]); }

  int size() const
  {
    int n = 0;
    for (int i = 0; i < NSECTORS; i++) {
      n += mTimeSlewing[i]->size();
    }
    return n;
  }

  int getSize(int sector) const { return mTimeSlewing[sector]->size(); }

  int getStartIndexForChannel(int sector, int channel) const { return (*(mChannelStart[sector]))[channel]; }
  int getStopIndexForChannel(int sector, int channel) const { return channel != NCHANNELXSECTOR - 1 ? (*(mChannelStart[sector]))[channel + 1] : getSize(sector) - 1; }
  float getFractionUnderPeak(int sector, int channel) const { return (*(mFractionUnderPeak[sector]))[channel]; }
  float getSigmaPeak(int sector, int channel) const { return (*(mSigmaPeak[sector]))[channel]; }
  float getFractionUnderPeak(int channel) const
  {
    int sector = channel / NCHANNELXSECTOR;
    int channelInSector = channel % NCHANNELXSECTOR;
    return getFractionUnderPeak(sector, channelInSector);
  }
  float getSigmaPeak(int channel) const
  {
    int sector = channel / NCHANNELXSECTOR;
    int channelInSector = channel % NCHANNELXSECTOR;
    return getSigmaPeak(sector, channelInSector);
  }
  void setFractionUnderPeak(int sector, int channel, float value) { (*(mFractionUnderPeak[sector]))[channel] = value; }
  void setSigmaPeak(int sector, int channel, float value) { (*(mSigmaPeak[sector]))[channel] = value; }

  bool isProblematic(int channel)
  {
    int sector = channel / NCHANNELXSECTOR;
    int channelInSector = channel % NCHANNELXSECTOR;
    return (getFractionUnderPeak(sector, channelInSector) < 0);
  }

  CalibTimeSlewingParamTOF& operator+=(const CalibTimeSlewingParamTOF& other);
  void bind();

  long getStartValidity() const { return mStartValidity; }
  long getEndValidity() const { return mEndValidity; }

  void setStartValidity(long validity) { mStartValidity = validity; }
  void setEndValidity(long validity) { mEndValidity = validity; }

 private:
  std::array<int, NCHANNELXSECTOR> mChannelStartSec0;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec0;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec0;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec0;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec0;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec1;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec1;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec1;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec1;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec1;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec2;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec2;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec2;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec2;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec2;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec3;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec3;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec3;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec3;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec3;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec4;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec4;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec4;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec4;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec4;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec5;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec5;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec5;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec5;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec5;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec6;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec6;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec6;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec6;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec6;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec7;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec7;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec7;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec7;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec7;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec8;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec8;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec8;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec8;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec8;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec9;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec9;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec9;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec9;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec9;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec10;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec10;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec10;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec10;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec10;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec11;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec11;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec11;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec11;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec11;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec12;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec12;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec12;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec12;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec12;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec13;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec13;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec13;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec13;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec13;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec14;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec14;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec14;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec14;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec14;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec15;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec15;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec15;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec15;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec15;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec16;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec16;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec16;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec16;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec16;

  std::array<int, NCHANNELXSECTOR> mChannelStartSec17;
  std::array<float, NCHANNELXSECTOR> mGlobalOffsetSec17;
  std::vector<std::pair<unsigned short, short>> mTimeSlewingSec17;
  std::array<float, NCHANNELXSECTOR> mFractionUnderPeakSec17;
  std::array<float, NCHANNELXSECTOR> mSigmaPeakSec17;

  // TOF channel calibrations
  std::array<int, NCHANNELXSECTOR>* mChannelStart[NSECTORS] = {&mChannelStartSec0, &mChannelStartSec1, &mChannelStartSec2, &mChannelStartSec3, &mChannelStartSec4, &mChannelStartSec5, &mChannelStartSec6, &mChannelStartSec7, &mChannelStartSec8, &mChannelStartSec9, &mChannelStartSec10, &mChannelStartSec11, &mChannelStartSec12, &mChannelStartSec13, &mChannelStartSec14, &mChannelStartSec15, &mChannelStartSec16, &mChannelStartSec17};                                                                                                  //! array with the index of the first element of a channel in the time slewing vector (per sector)
  std::array<float, NCHANNELXSECTOR>* mGlobalOffset[NSECTORS] = {&mGlobalOffsetSec0, &mGlobalOffsetSec1, &mGlobalOffsetSec2, &mGlobalOffsetSec3, &mGlobalOffsetSec4, &mGlobalOffsetSec5, &mGlobalOffsetSec6, &mGlobalOffsetSec7, &mGlobalOffsetSec8, &mGlobalOffsetSec9, &mGlobalOffsetSec10, &mGlobalOffsetSec11, &mGlobalOffsetSec12, &mGlobalOffsetSec13, &mGlobalOffsetSec14, &mGlobalOffsetSec15, &mGlobalOffsetSec16, &mGlobalOffsetSec17};                                                                                                //! array with the sigma of the peak
  std::vector<std::pair<unsigned short, short>>* mTimeSlewing[NSECTORS] = {&mTimeSlewingSec0, &mTimeSlewingSec1, &mTimeSlewingSec2, &mTimeSlewingSec3, &mTimeSlewingSec4, &mTimeSlewingSec5, &mTimeSlewingSec6, &mTimeSlewingSec7, &mTimeSlewingSec8, &mTimeSlewingSec9, &mTimeSlewingSec10, &mTimeSlewingSec11, &mTimeSlewingSec12, &mTimeSlewingSec13, &mTimeSlewingSec14, &mTimeSlewingSec15, &mTimeSlewingSec16, &mTimeSlewingSec17};                                                                                                        //! array of sector vectors; first element of the pair is TOT (in ps), second is t-texp_pi (in ps)
  std::array<float, NCHANNELXSECTOR>* mFractionUnderPeak[NSECTORS] = {&mFractionUnderPeakSec0, &mFractionUnderPeakSec1, &mFractionUnderPeakSec2, &mFractionUnderPeakSec3, &mFractionUnderPeakSec4, &mFractionUnderPeakSec5, &mFractionUnderPeakSec6, &mFractionUnderPeakSec7, &mFractionUnderPeakSec8, &mFractionUnderPeakSec9, &mFractionUnderPeakSec10, &mFractionUnderPeakSec11, &mFractionUnderPeakSec12, &mFractionUnderPeakSec13, &mFractionUnderPeakSec14, &mFractionUnderPeakSec15, &mFractionUnderPeakSec16, &mFractionUnderPeakSec17}; //! array with the fraction of entries below the peak
  std::array<float, NCHANNELXSECTOR>* mSigmaPeak[NSECTORS] = {&mSigmaPeakSec0, &mSigmaPeakSec1, &mSigmaPeakSec2, &mSigmaPeakSec3, &mSigmaPeakSec4, &mSigmaPeakSec5, &mSigmaPeakSec6, &mSigmaPeakSec7, &mSigmaPeakSec8, &mSigmaPeakSec9, &mSigmaPeakSec10, &mSigmaPeakSec11, &mSigmaPeakSec12, &mSigmaPeakSec13, &mSigmaPeakSec14, &mSigmaPeakSec15, &mSigmaPeakSec16, &mSigmaPeakSec17};                                                                                                                                                         //! array with the sigma of the peak

  long mStartValidity = 0; ///< start validity of the object when put in CCDB
  long mEndValidity = 0;   ///< end validity of the object when put in CCDB

  ClassDefNV(CalibTimeSlewingParamTOF, 4); // class for TOF time slewing params
};
} // namespace dataformats
} // namespace o2
#endif
