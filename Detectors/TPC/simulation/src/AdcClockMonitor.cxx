/// \file AdcClockMonitor.cxx
/// \author Sebastian Klewin

#include "TPCSimulation/AdcClockMonitor.h"

using namespace AliceO2::TPC;

AdcClockMonitor::AdcClockMonitor()
  : mPrevSequence(0)
  , mTransition0(0)
  , mTransition1(0)
  , mSequenceCompleted(false)
  , mSequencePosition(0)
  , mState(state::error)
{}

AdcClockMonitor::~AdcClockMonitor()
{}

void AdcClockMonitor::reset()
{
  mState = state::error;
  mSequencePosition = 0;
}

int AdcClockMonitor::addSequence(char seq)
{
  seq = seq & 0xF;

  switch(mState) {
    case state::error:
      mSequenceCompleted = false;
      if (seq == 0x1) {
        mState = state::locked;
        mSequencePosition = 0;
        mTransition0 = 0xE;
        mTransition1 = 0x1;
      } else if (seq == 0x3) {
        mState = state::locked;
        mSequencePosition = 0;
        mTransition0 = 0xC;
        mTransition1 = 0x3;
      } else if (seq == 0x7) {
        mState = state::locked;
        mSequencePosition = 0;
        mTransition0 = 0x8;
        mTransition1 = 0x7;
      } else if (seq == 0xF) {
        if (mPrevSequence == 0) 
        {
          mState = state::locked;
          mSequencePosition = 0;
          mTransition0 = 0x0;
          mTransition1 = 0xF;
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
      } else if (mSequencePosition == 3) {
        if (seq != mTransition0) mState = state::error;
        ++mSequencePosition;
      } else if (
          mSequencePosition == 4 ||
          mSequencePosition == 5 ||
          mSequencePosition == 6) {
        if (seq != 0x0) mState = state::error;
        mSequencePosition++;
      } else if (mSequencePosition == 7) {
        if (seq != mTransition1) mState = state::error;
        ++mSequencePosition;
        mSequenceCompleted = true;
      } else {
        mState = state::error;
      }

      break;
  }
  mSequencePosition %= 8;
  mPrevSequence = seq & 0xF;

  return getState();
}
