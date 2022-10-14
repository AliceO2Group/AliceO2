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

#ifndef ALICEO2_TOF_UTILS_H_
#define ALICEO2_TOF_UTILS_H_

#include <iosfwd>
#include "Rtypes.h"
#include "TOFBase/Geo.h"
#include "TOFBase/EventTimeMaker.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"
#include <vector>

#include <TF1.h>
#include <TChain.h>
#include <TH2F.h>

namespace o2
{
namespace tof
{
/// \class Utils
/// \brief TOF utils

template <typename trackType>
bool filterCalib(const o2::dataformats::CalibInfoTOF& tr)
{
  return true;
} // accept all

class Utils
{
 public:
  Utils() = default;

  static bool hasFillScheme();
  static int getNinteractionBC();
  static void addBC(float toftime, bool subLatency = false);
  static void addBC(double toftime, bool subLatency = false) { addBC(float(toftime), subLatency); }
  static void addInteractionBC(int bc, bool fromCollisonCotext = false);
  static int getInteractionBC(int ibc) { return mFillScheme[ibc]; }
  static double subtractInteractionBC(double time, int& mask, bool subLatency = false);
  static float subtractInteractionBC(float time, int& mask, bool subLatency = false);
  static void init();
  static void printFillScheme();
  static void addCalibTrack(float time);
  static void computeLHCphase();

  // info can be tuned
  static float mEventTimeSpread;
  static float mEtaMin;
  static float mEtaMax;
  static float mLHCPhase;

  static int addMaskBC(int mask, int channel);
  static int getMaxUsed();
  static int getMaxUsedChannel(int channel);
  static int extractNewTimeSlewing(const dataformats::CalibTimeSlewingParamTOF* oldTS, dataformats::CalibTimeSlewingParamTOF* newTS);
  static void fitTimeSlewing(int sector, const dataformats::CalibTimeSlewingParamTOF* oldTS, dataformats::CalibTimeSlewingParamTOF* newTS);
  static void fitChannelsTS(int chStart, const dataformats::CalibTimeSlewingParamTOF* oldTS, dataformats::CalibTimeSlewingParamTOF* newTS);
  static int fitSingleChannel(int ch, TH2F* h, const dataformats::CalibTimeSlewingParamTOF* oldTS, dataformats::CalibTimeSlewingParamTOF* newTS);

 private:
  static std::vector<int> mFillScheme;
  static int mBCmult[o2::constants::lhc::LHCMaxBunches];
  static int mNautodet;
  static int mMaxBC;
  static bool mIsInit;

  // for LHCphase from calib infos
  static constexpr int NTRACKS_REQUESTED = 1000;
  static int mNCalibTracks;
  static o2::dataformats::CalibInfoTOF mCalibTracks[NTRACKS_REQUESTED];
  static int mNsample;
  static int mIsample;
  static float mPhases[100];
  static uint64_t mMaskBC[16];
  static uint64_t mMaskBCUsed[16];
  static int mMaskBCchan[o2::tof::Geo::NCHANNELS][16];
  static int mMaskBCchanUsed[o2::tof::Geo::NCHANNELS][16];

  static TChain* mTreeFit;
  static std::vector<dataformats::CalibInfoTOF> mVectC;
  static std::vector<dataformats::CalibInfoTOF>* mPvectC;
  static const int NCHPERBUNCH = Geo::NCHANNELS / Geo::NSECTORS / 16;
  static const int NMINTOFIT = 300;
  static int mNfits;

  ClassDefNV(Utils, 1);
};

} // namespace tof
} // namespace o2
#endif
