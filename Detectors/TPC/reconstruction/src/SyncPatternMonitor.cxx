/// \file SyncPatternMonitor.cxx
/// \author Sebastian Klewin

#include "TPCReconstruction/SyncPatternMonitor.h"

using namespace o2::TPC;
constexpr std::array<short,32> SyncPatternMonitor::SYNC_PATTERN;

SyncPatternMonitor::SyncPatternMonitor()
  : SyncPatternMonitor(-1,-1)
{}

SyncPatternMonitor::SyncPatternMonitor(int sampa, int lowHigh)
  : mPatternFound(false)
  , mPosition(SYNC_START)
  , mHwWithPattern(-1)
  , mSampa(sampa)
  , mLowHigh(lowHigh)
  , mCheckedWords(0)
{}

SyncPatternMonitor::SyncPatternMonitor(const SyncPatternMonitor& other)
  = default;

void SyncPatternMonitor::reset()
{
  mPatternFound = false;
  mPosition = SYNC_START;
  mHwWithPattern = -1;
  mCheckedWords = 0;
  LOG(INFO) << "Sync pattern monitoring for SAMPA " << mSampa << " (" << ((mLowHigh == 0) ? "low" : "high") << " bits) "
    << "was resetted" << FairLogger::endl;
}


