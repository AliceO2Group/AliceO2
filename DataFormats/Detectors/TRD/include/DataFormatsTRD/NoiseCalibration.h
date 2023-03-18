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

#ifndef ALICEO2_TRD_NOISECALIBRATION_H
#define ALICEO2_TRD_NOISECALIBRATION_H

#include "Rtypes.h"
#include <bitset>
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Digit.h"

namespace o2
{
namespace trd
{

/// \class NoiseStatusMCM
/// \brief Simple noise status bit for each MCM of the TRD
/// \author Ole Schmidt

class ChannelInfo
{
 public:
  ChannelInfo() = default;

  bool isDummy() const { return mNEntries == 0; }
  float getMean() const { return mMean; }
  float getRMS() const { return mRMS; }
  uint32_t getEntries() const { return mNEntries; }

  void setMean(float mean) { mMean = mean; }
  void setRMS(float rms) { mRMS = rms; }
  void setNentries(uint32_t n) { mNEntries = n; }

 private:
  float mMean{0.f};
  float mRMS{0.f};
  uint32_t mNEntries{0};
  ClassDefNV(ChannelInfo, 1);
};

class ChannelInfoContainer
{
 public:
  ChannelInfoContainer() { mData.resize(constants::NCHANNELSTOTAL); }
  ChannelInfo& getChannel(int index) { return mData[index]; }
  ChannelInfo getChannel(int index) const { return mData[index]; }

  const std::vector<ChannelInfo>& getData() const { return mData; }

 private:
  std::vector<ChannelInfo> mData{};
  ClassDefNV(ChannelInfoContainer, 1);
};

class NoiseStatusMCM
{

 public:
  NoiseStatusMCM() = default;

  // convert global MCM index into HC, ROB and MCM number
  static constexpr void convertMcmIdxGlb(int mcmGlb, int& hcid, int& rob, int& mcm)
  {
    hcid = mcmGlb / constants::NMCMHCMAX;
    int side = (hcid % 2) ? 1 : 0;
    rob = ((mcmGlb % constants::NMCMHCMAX) / constants::NMCMROB) * 2 + side;
    mcm = (mcmGlb % constants::NMCMHCMAX) % constants::NMCMROB;
  }
  // convert HC, ROB and MCM number into a global MCM index
  static constexpr int getMcmIdxGlb(int hcid, int rob, int mcm) { return hcid * constants::NMCMHCMAX + (rob / 2) * constants::NMCMROB + mcm; }

  // setters
  void setIsNoisy(int hcid, int rob, int mcm) { mNoiseFlag.set(getMcmIdxGlb(hcid, rob, mcm)); }
  void setIsNoisy(int mcmIdxGlb) { mNoiseFlag.set(mcmIdxGlb); }

  // getters
  bool getIsNoisy(int hcid, int rob, int mcm) const { return mNoiseFlag.test(getMcmIdxGlb(hcid, rob, mcm)); }
  bool getIsNoisy(int mcmIdxGlb) const { return mNoiseFlag.test(mcmIdxGlb); }
  auto getNumberOfNoisyMCMs() const { return mNoiseFlag.count(); }
  bool isTrackletFromNoisyMCM(const Tracklet64& trklt) const { return getIsNoisy(trklt.getHCID(), trklt.getROB(), trklt.getMCM()); }
  bool isDigitFromNoisyMCM(const Digit& d) const { return getIsNoisy(d.getHCId(), d.getROB(), d.getMCM()); }

 private:
  std::bitset<constants::MAXHALFCHAMBER * constants::NMCMHCMAX> mNoiseFlag{};

  ClassDefNV(NoiseStatusMCM, 1);
};

} // namespace trd
} // namespace o2

#endif // ALICEO2_TRD_NOISECALIBRATION_H
