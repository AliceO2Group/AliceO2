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

#include <TH1.h>
#include <TH2.h>
#include "ZDCBase/Constants.h"
#include "ZDCSimulation/ZDCSimParam.h"
#include "DataFormatsZDC/RawEventData.h"

#ifndef ALICEO2_ZDC_DUMPRAW_H_
#define ALICEO2_ZDC_DUMPRAW_H_

namespace o2
{
namespace zdc
{
class DumpRaw
{
 public:
  DumpRaw() = default;
  void init();
  int process(const EventData& ev);
  int process(const EventChData& ch);
  int processWord(const uint32_t* word);
  int getHPos(uint32_t board, uint32_t ch);
  void write();
  void setVerbosity(int v)
  {
    mVerbosity = v;
  }
  int getVerbosity() const { return mVerbosity; }

 private:
  void setStat(TH1* h);
  void setModuleLabel(TH1* h);
  void setTriggerYLabel(TH2* h);
  int mVerbosity = 1;
  std::unique_ptr<TH2> mTransmitted = nullptr;
  std::unique_ptr<TH2> mFired = nullptr;
  std::unique_ptr<TH2> mBits = nullptr;
  std::unique_ptr<TH2> mBitsH = nullptr;
  std::unique_ptr<TH1> mLoss = nullptr;
  std::unique_ptr<TH1> mOve = nullptr;
  std::unique_ptr<TH1> mBaseline[NDigiChannels] = {nullptr};
  std::unique_ptr<TH1> mCounts[NDigiChannels] = {nullptr};
  std::unique_ptr<TH2> mSignalA[NDigiChannels] = {nullptr};
  std::unique_ptr<TH2> mSignalT[NDigiChannels] = {nullptr};
  std::unique_ptr<TH2> mBunchA[NDigiChannels] = {nullptr};
  std::unique_ptr<TH2> mBunchT[NDigiChannels] = {nullptr};
  EventChData mCh;
};
} // namespace zdc
} // namespace o2
#endif
