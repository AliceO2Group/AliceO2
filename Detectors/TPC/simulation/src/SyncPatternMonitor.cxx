/// \file SyncPatternMonitor.cxx
/// \author Sebastian Klewin

#include "TPCSimulation/SyncPatternMonitor.h"

using namespace AliceO2::TPC;
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
  : mPatternFound(other.mPatternFound)
  , mPosition(other.mPosition)
  , mHwWithPattern(other.mHwWithPattern)
  , mSampa(other.mSampa)
  , mLowHigh(other.mLowHigh)
  , mCheckedWords(other.mCheckedWords)
{}

SyncPatternMonitor::~SyncPatternMonitor()
{}

void SyncPatternMonitor::reset()
{
  mPatternFound = false;
  mPosition = SYNC_START;
  mHwWithPattern = -1;
  mCheckedWords = 0;
  LOG(INFO) << "Sync pattern monitoring for SAMPA " << mSampa << " (" << ((mLowHigh == 0) ? "low" : "high") << " bits) "
    << "was resetted" << FairLogger::endl;
}

short SyncPatternMonitor::addSequence(const short hw0, const short hw1, const short hw2, const short hw3)
{
//  std::cout << std::hex
//    << "0x" << std::setw(2) << (int) hw0 << " "
//    << "0x" << std::setw(2) << (int) hw1 << " "
//    << "0x" << std::setw(2) << (int) hw2 << " "
//    << "0x" << std::setw(2) << (int) hw3 << " ";

  checkWord(hw0,1);
  checkWord(hw1,2);
  checkWord(hw2,3);
  checkWord(hw3,0);

//  std::cout << std::dec << mPosition << " " << mPatternFound << " " << mHwWithPattern << " " << getPosition() << std::endl;

  return getPosition();
}

