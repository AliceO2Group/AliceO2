/// \file AdcClockMonitor.cxx
/// \author Sebastian Klewin

#include "TPCReconstruction/AdcClockMonitor.h"

using namespace o2::TPC;

AdcClockMonitor::AdcClockMonitor()
  : AdcClockMonitor(-1)
{}

AdcClockMonitor::AdcClockMonitor(int sampa)
  : mSampa(sampa)
  , mPrevSequence(0)
  , mTransition0(0)
  , mTransition1(0)
  , mSequenceCompleted(false)
  , mSequencePosition(0)
  , mState(state::error)
{}

AdcClockMonitor::AdcClockMonitor(const AdcClockMonitor& other)
  = default;


AdcClockMonitor::~AdcClockMonitor()
= default;

void AdcClockMonitor::reset()
{
  mPrevSequence = 0;
  mSequencePosition = 0;
  mState = state::error;
  LOG(INFO) << "ADC clock monitoring for SAMPA " << mSampa << " was resetted" << FairLogger::endl; 
}

int AdcClockMonitor::addSequence(const short seq)
{
//  seq = seq & 0xF;
//  std::cout << "SAMPA " << mSampa << ": " << std::hex << "0x" << std::setw(1) << (int)seq << std::dec << std::endl;

  switch(mState) {
    case state::error:
      mSequenceCompleted = false;
      if (seq == 0x1) {
        mState = state::locked;
        mSequencePosition = 0;
        mTransition0 = 0xE;
        mTransition1 = 0x1;
        break;
      } 
      if (seq == 0x3) {
        mState = state::locked;
        mSequencePosition = 0;
        mTransition0 = 0xC;
        mTransition1 = 0x3;
        break;
      } 
      if (seq == 0x7) {
        mState = state::locked;
        mSequencePosition = 0;
        mTransition0 = 0x8;
        mTransition1 = 0x7;
        break;
      } 
      if (seq == 0xF) {
        if (mPrevSequence == 0) 
        {
          mState = state::locked;
          mSequencePosition = 0;
          mTransition0 = 0x0;
          mTransition1 = 0xF;
          break;
        }
      }
      break;
    case state::locked:
      if (
          mSequencePosition == 0 ||
          mSequencePosition == 1 ||
          mSequencePosition == 2) 
      {
        if (seq != 0xF) mState = state::error;
        ++mSequencePosition;
        break;
      }
      if (mSequencePosition == 3) {
        if (seq != mTransition0) mState = state::error;
        ++mSequencePosition;
        break;
      }
      if (
          mSequencePosition == 4 ||
          mSequencePosition == 5 ||
          mSequencePosition == 6) {
        if (seq != 0x0) mState = state::error;
        mSequencePosition++;
        break;
      }
      if (mSequencePosition == 7) {
        if (seq != mTransition1) mState = state::error;
        ++mSequencePosition;
        mSequenceCompleted = true;
        break;
      }

      mState = state::error;
      break;
  }
  mSequencePosition %= 8;
  mPrevSequence = seq & 0xF;

  return getState();
}
