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

  CalibTimeSlewingParamTOF(const CalibTimeSlewingParamTOF& source) = default;

  CalibTimeSlewingParamTOF& operator=(const CalibTimeSlewingParamTOF& source) = default;

  float getChannelOffset(int channel) const;

  float evalTimeSlewing(int channel, float tot) const;

  void addTimeSlewingInfo(int channel, float tot, float time);

  bool updateOffsetInfo(int channel, float residualOffset);

  const std::vector<std::pair<unsigned short, short>>& getVector(int sector) const { return mTimeSlewing[sector]; }

  int size() const
  {
    int n = 0;
    for (int i = 0; i < NSECTORS; i++) {
      n += (mTimeSlewing[i]).size();
    }
    return n;
  }

  int getSize(int sector) const { return mTimeSlewing[sector].size(); }

  int getStartIndexForChannel(int sector, int channel) const { return mChannelStart[sector][channel]; }
  int getStopIndexForChannel(int sector, int channel) const { return channel != NCHANNELXSECTOR - 1 ? mChannelStart[sector][channel + 1] : getSize(sector) - 1; }
  float getFractionUnderPeak(int sector, int channel) const { return mFractionUnderPeak[sector][channel]; }
  float getSigmaPeak(int sector, int channel) const { return mSigmaPeak[sector][channel]; }
  float getFractionUnderPeak(int channel) const
  {
    int sector = channel / NCHANNELXSECTOR;
    int channelInSector = channel % NCHANNELXSECTOR;
    return getFractionUnderPeak(sector, channelInSector);
  }

  void setFractionUnderPeak(int sector, int channel, float value) { mFractionUnderPeak[sector][channel] = value; }
  void setSigmaPeak(int sector, int channel, float value) { mSigmaPeak[sector][channel] = value; }

  bool isProblematic(int channel)
  {
    int sector = channel / NCHANNELXSECTOR;
    int channelInSector = channel % NCHANNELXSECTOR;
    return (getFractionUnderPeak(sector, channelInSector) < 0);
  }

  CalibTimeSlewingParamTOF& operator+=(const CalibTimeSlewingParamTOF& other);

 private:
  // TOF channel calibrations
  std::array<std::array<int, NCHANNELXSECTOR>, NSECTORS> mChannelStart;             ///< array with the index of the first element of a channel in the time slewing vector (per sector)
  std::array<std::vector<std::pair<unsigned short, short>>, NSECTORS> mTimeSlewing; ///< array of sector vectors; first element of the pair is TOT (in ps), second is t-texp_pi (in ps)
  std::array<std::array<float, NCHANNELXSECTOR>, NSECTORS> mFractionUnderPeak;      ///< array with the fraction of entries below the peak
  std::array<std::array<float, NCHANNELXSECTOR>, NSECTORS> mSigmaPeak;              ///< array with the sigma of the peak

  ClassDefNV(CalibTimeSlewingParamTOF, 1); // class for TOF time slewing params
};
} // namespace dataformats
} // namespace o2
#endif
