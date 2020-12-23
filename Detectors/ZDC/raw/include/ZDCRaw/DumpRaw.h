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
  TH1* mBaseline[NDigiChannels] = {0};
  TH1* mCounts[NDigiChannels] = {0};
  TH2* mSignal[NDigiChannels] = {0};
  TH2* mBunch[NDigiChannels] = {0};
  EventChData mCh;
};
} // namespace zdc
} // namespace o2

#endif
