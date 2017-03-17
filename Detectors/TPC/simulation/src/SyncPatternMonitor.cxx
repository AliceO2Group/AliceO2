/// \file SyncPatternMonitor.cxx
/// \author Sebastian Klewin

#include "TPCSimulation/SyncPatternMonitor.h"

using namespace AliceO2::TPC;

SyncPatternMonitor::SyncPatternMonitor()
  : SyncPatternMonitor(-1,-1)
{}

SyncPatternMonitor::SyncPatternMonitor(int sampa, int lowHigh)
  : mCurrentState(state::lookForSeq0)
  , mPatternFound(false)
  , mPosition(-1)
  , mTempPosition(-1)
  , mLastHw0(0)
  , mLastHw1(0)
  , mLastHw2(0)
  , mLastHw3(0)
  , mSampa(sampa)
  , mLowHigh(lowHigh)
{}

SyncPatternMonitor::SyncPatternMonitor(const SyncPatternMonitor& other)
  : mCurrentState(other.mCurrentState)
  , mPatternFound(other.mPatternFound)
  , mPosition(other.mPosition)
  , mTempPosition(other.mTempPosition)
  , mLastHw0(other.mLastHw0)
  , mLastHw1(other.mLastHw1)
  , mLastHw2(other.mLastHw2)
  , mLastHw3(other.mLastHw3)
  , mSampa(other.mSampa)
  , mLowHigh(other.mLowHigh)
{}

SyncPatternMonitor::~SyncPatternMonitor()
{}

void SyncPatternMonitor::reset()
{
  mCurrentState = state::lookForSeq0;
  mPatternFound = false;
  mPosition = -1;
  mTempPosition = -1;
  mLastHw0 = 0;
  mLastHw1 = 0;
  mLastHw2 = 0;
  mLastHw3 = 0;
  LOG(INFO) << "Sync pattern monitoring for SAMPA " << mSampa << " (" << ((mLowHigh == 0) ? "low" : "high") << " bits) "
    << "was resetted" << FairLogger::endl;
}

int SyncPatternMonitor::addSequence(const short& hw0, const short& hw1, const short& hw2, const short& hw3)
{
  int iLastPosition = mPosition;
//  hw0 %= 0x1F;
//  hw1 %= 0x1F;
//  hw2 %= 0x1F;
//  hw3 %= 0x1F;
 
//  std::cout <<  (int)hw0 << " " << (int)hw1 << " " << (int)hw2 << " " << (int)hw3 << std::endl;
  short iCheckPos0, iCheckPos1, iCheckPos2, iCheckPos3;

  if (mCurrentState != state::lookForSeq0) 
  {
    switch(mTempPosition)
    {
      case 0:
        iCheckPos0 = hw0;
        iCheckPos1 = hw1;
        iCheckPos2 = hw2;
        iCheckPos3 = hw3;
        break;

      case 1:
        iCheckPos0 = mLastHw1;
        iCheckPos1 = mLastHw2;
        iCheckPos2 = mLastHw3;
        iCheckPos3 = hw0;
        break;

      case 2:
        iCheckPos0 = mLastHw2;
        iCheckPos1 = mLastHw3;
        iCheckPos2 = hw0;
        iCheckPos3 = hw1;
        break;

      case 3:
        iCheckPos0 = mLastHw3;
        iCheckPos1 = hw0;
        iCheckPos2 = hw1;
        iCheckPos3 = hw2;
        break;
      default:
          LOG(ERROR) << "SAMPA " << mSampa << " (" << ((mLowHigh == 0) ? "low" : "high") << " bits): "
                     << "Position " << mTempPosition << " not defined" << FairLogger::endl;
    }
  }

  switch (mCurrentState) 
  {
    case state::lookForSeq0:
      if (/*hw0 == PATTERN_A &&*/ hw1 == PATTERN_A &&
          hw2 == PATTERN_B && hw3 == PATTERN_B) {
        incState();
        mTempPosition = 0;
      } else if (/*mLastHw1 == PATTERN_A &&*/ mLastHw2 == PATTERN_A &&
                 mLastHw3 == PATTERN_B && hw0      == PATTERN_B)  {
        incState();
        mTempPosition = 1;
      } else if (/*mLastHw2 == PATTERN_A &&*/ mLastHw3 == PATTERN_A &&
                 hw0      == PATTERN_B && hw1      == PATTERN_B)  {
        incState();
        mTempPosition = 2;
      } else if (/*mLastHw3 == PATTERN_A &&*/ hw0 == PATTERN_A &&
                 hw1      == PATTERN_B && hw2 == PATTERN_B)  {
        incState();
        mTempPosition = 3;
      }
      break;

    case state::lookForSeq1: case state::lookForSeq2: case state::lookForSeq3:
      if (iCheckPos0 == PATTERN_A && iCheckPos1 == PATTERN_A &&
          iCheckPos2 == PATTERN_B && iCheckPos3 == PATTERN_B) {
        incState();
      } else {
        mCurrentState = state::lookForSeq0;
      }
      break;

    case state::lookForSeq4: case state::lookForSeq6:
      if (iCheckPos0 == PATTERN_A && iCheckPos1 == PATTERN_A &&
          iCheckPos2 == PATTERN_A && iCheckPos3 == PATTERN_A) {
        incState();
      } else {
        mCurrentState = state::lookForSeq0;
      }
      break;

    case state::lookForSeq5: case state::lookForSeq7:
      if (iCheckPos0 == PATTERN_B && iCheckPos1 == PATTERN_B &&
          iCheckPos2 == PATTERN_B && iCheckPos3 == PATTERN_B) {
        if (mCurrentState == state::lookForSeq7) 
        {
          LOG(INFO) << "SAMPA " << mSampa << " (" << ((mLowHigh == 0) ? "low" : "high") << " bits): "
            << "Synchronization pattern found, started at position " << mTempPosition << FairLogger::endl;
          if (mPatternFound == true) {
            LOG(WARNING) << "SAMPA " << mSampa << " (" << ((mLowHigh == 0) ? "low" : "high") << " bits): "
                         << "Synchronization was already found" << FairLogger::endl;
          }
          mPatternFound = true;
          mPosition = mTempPosition;
        }
        incState();
      } else {
        mCurrentState = state::lookForSeq0;
      }
      break;
  }

  if ((iLastPosition != mPosition) && (iLastPosition != -1)) {
    LOG(WARNING) << "SAMPA " << mSampa << " (" << ((mLowHigh == 0) ? "low" : "high") << " bits): "
                 << "Position of synchronization pattern changed from " << iLastPosition << " to " << mPosition << FairLogger::endl;
  }

//  printState(mCurrentState);

  mLastHw0 = hw0;
  mLastHw1 = hw1;
  mLastHw2 = hw2;
  mLastHw3 = hw3;
  return getPosition();
}

void SyncPatternMonitor::incState()
{
  switch(mCurrentState) 
  {
    case state::lookForSeq0: mCurrentState = state::lookForSeq1; break;
    case state::lookForSeq1: mCurrentState = state::lookForSeq2; break;
    case state::lookForSeq2: mCurrentState = state::lookForSeq3; break;
    case state::lookForSeq3: mCurrentState = state::lookForSeq4; break;
    case state::lookForSeq4: mCurrentState = state::lookForSeq5; break;
    case state::lookForSeq5: mCurrentState = state::lookForSeq6; break;
    case state::lookForSeq6: mCurrentState = state::lookForSeq7; break;
    case state::lookForSeq7: mCurrentState = state::lookForSeq0; break;
  }
}

void SyncPatternMonitor::printState(state stateToPrint)
{
  switch(stateToPrint) 
  {
    case state::lookForSeq0: std::cout << "lookForSeq0" << std::endl; break;
    case state::lookForSeq1: std::cout << "lookForSeq1" << std::endl; break;
    case state::lookForSeq2: std::cout << "lookForSeq2" << std::endl; break;
    case state::lookForSeq3: std::cout << "lookForSeq3" << std::endl; break;
    case state::lookForSeq4: std::cout << "lookForSeq4" << std::endl; break;
    case state::lookForSeq5: std::cout << "lookForSeq5" << std::endl; break;
    case state::lookForSeq6: std::cout << "lookForSeq6" << std::endl; break;
    case state::lookForSeq7: std::cout << "lookForSeq7" << std::endl; break;
  }
}
