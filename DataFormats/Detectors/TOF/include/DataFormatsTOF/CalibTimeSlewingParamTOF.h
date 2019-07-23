// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibTimeSlewingParamTOF.h
/// \brief Class to store the output of the matching to TOF for calibration

#ifndef ALICEO2_CALIBTIMESLEWINGPARAMTOF_H
#define ALICEO2_CALIBTIMESLEWINGPARAMTOF_H

#include <vector>

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

  float evalTimeSlewing(int channel, float tot) const;

  void addTimeSlewingInfo(int channel, float tot, float time);

  const std::vector<std::pair<float, float>>* getVector(int sector) const { return mTimeSlewing[sector]; }

  int size() const
  {
    int n = 0;
    for (int i = 0; i < NSECTORS; i++)
      n += (*(mTimeSlewing[i])).size();
    return n;
  }

  int getSize(int sector) const { return (*(mTimeSlewing[sector])).size(); }

  int getStartTimeStamp(int sector, int channel) const { return mChannelStart[sector][channel]; }
  float getFractionUnderPeak(int sector, int channel) const { return mFractionUnderPeak[sector][channel]; }
  float getSigmaPeak(int sector, int channel) const { return mSigmaPeak[sector][channel]; }

  void setFractionUnderPeak(int sector, int channel, float value) { mFractionUnderPeak[sector][channel] = value; }
  void setSigmaPeak(int sector, int channel, float value) { mSigmaPeak[sector][channel] = value; }

  CalibTimeSlewingParamTOF& operator+=(const CalibTimeSlewingParamTOF& other);

 private:
  // TOF channel calibrations
  int mChannelStart[NSECTORS][NCHANNELXSECTOR];           ///< array with the index of the first element of a channel in the time slewing vector (per sector)
  std::vector<std::pair<float, float>>* mTimeSlewing[18]; //! pointers to the sector vectors

  std::vector<std::pair<float, float>> mTimeSlewingSec00; ///< timeslweing correction <tot,time> sector 0
  std::vector<std::pair<float, float>> mTimeSlewingSec01; ///< timeslweing correction <tot,time> sector 1
  std::vector<std::pair<float, float>> mTimeSlewingSec02; ///< timeslweing correction <tot,time> sector 2
  std::vector<std::pair<float, float>> mTimeSlewingSec03; ///< timeslweing correction <tot,time> sector 3
  std::vector<std::pair<float, float>> mTimeSlewingSec04; ///< timeslweing correction <tot,time> sector 4
  std::vector<std::pair<float, float>> mTimeSlewingSec05; ///< timeslweing correction <tot,time> sector 5
  std::vector<std::pair<float, float>> mTimeSlewingSec06; ///< timeslweing correction <tot,time> sector 6
  std::vector<std::pair<float, float>> mTimeSlewingSec07; ///< timeslweing correction >tot,time> sector 7
  std::vector<std::pair<float, float>> mTimeSlewingSec08; ///< timeslweing correction <tot,time> sector 8
  std::vector<std::pair<float, float>> mTimeSlewingSec09; ///< timeslweing correction <tot,time> sector 9
  std::vector<std::pair<float, float>> mTimeSlewingSec10; ///< timeslweing correction <tot,time> sector 10
  std::vector<std::pair<float, float>> mTimeSlewingSec11; ///< timeslweing correction <tot,time> sector 11
  std::vector<std::pair<float, float>> mTimeSlewingSec12; ///< timeslweing correction <tot,time> sector 12
  std::vector<std::pair<float, float>> mTimeSlewingSec13; ///< timeslweing correction <tot,time> sector 13
  std::vector<std::pair<float, float>> mTimeSlewingSec14; ///< timeslweing correction <tot,time> sector 14
  std::vector<std::pair<float, float>> mTimeSlewingSec15; ///< timeslweing correction <tot,time> sector 15
  std::vector<std::pair<float, float>> mTimeSlewingSec16; ///< timeslweing correction <tot,time> sector 16
  std::vector<std::pair<float, float>> mTimeSlewingSec17; ///< timeslweing correction <tot,time> sector 17

  float mFractionUnderPeak[NSECTORS][NCHANNELXSECTOR]; ///< array with the fraction of entries below the peak
  float mSigmaPeak[NSECTORS][NCHANNELXSECTOR];         ///< array with the sigma of the peak

  //  ClassDefNV(CalibTimeSlewingParamTOF, 2); // class for TOF time slewing params
};
} // namespace dataformats
} // namespace o2
#endif
