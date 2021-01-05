// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  int processWord(const UInt_t* word);
  int getHPos(uint32_t board, uint32_t ch);
  void write();
  void setVerbosity(int v)
  {
    mVerbosity = v;
  }
  int getVerbosity() const { return mVerbosity; }

 private:
  void setStat(TH1* h);
  int mVerbosity = 1;
  TH1* mBaseline[NDigiChannels] = {nullptr};
  TH1* mCounts[NDigiChannels] = {nullptr};
  TH2* mSignal[NDigiChannels] = {nullptr};
  TH2* mBunch[NDigiChannels] = {nullptr};
  EventChData mCh;
};
} // namespace zdc
} // namespace o2

#endif
