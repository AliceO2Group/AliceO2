/// \file SyncPatternMonitor.h
/// \brief Class to monitor the data stream and detect synchronization patterns
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_SYNCPATTERNMONITOR_H_
#define ALICEO2_TPC_SYNCPATTERNMONITOR_H_

#include "FairLogger.h"
#include <iostream>

namespace AliceO2 {
namespace TPC {

/// \class SyncPatternMonitor
/// \brief Class to monitor the data stream and detect synchronization patterns

class SyncPatternMonitor {
  public:

    /// Default Constructor
    SyncPatternMonitor();

    /// Constructor
    SyncPatternMonitor(int sampa, int lowHigh);

    /// Copy Constructor
    SyncPatternMonitor(const SyncPatternMonitor& other);

    /// Destructor
    ~SyncPatternMonitor();

    /// Reset function to clear history
    void reset();

    /// Adds a sequence of 4 new half-words and looks for sync pattern
    /// @param hw0 1st (timewise) half word
    /// @param hw1 2nd (timewise) half word
    /// @param hw2 3th (timewise) half word
    /// @param hw3 4th (timewise) half word
    /// @return Position of first part of the synchronization pattern, -1 if no pattern was found
    int addSequence(const short& hw0, const short& hw1, const short& hw2, const short& hw3);

    /// Get position
    /// @return Position of first part of the synchronization pattern, -1 if no patter was found
    int getPosition() { return mPatternFound ? mPosition : -1; };

  private:

    const static short PATTERN_A = 0x15;
    const static short PATTERN_B = 0x0A;

    enum state {lookForSeq0, lookForSeq1, lookForSeq2, lookForSeq3,
                lookForSeq4, lookForSeq5, lookForSeq6, lookForSeq7};

    /// increments mCurrentState
    void incState();

    void printState(state stateToPrint);

    state mCurrentState;    ///< store current state
    bool mPatternFound;     ///< store whether pattern was already found
    int mPosition;          ///< position of last part of the pattern
    int mTempPosition;      ///< temporary postion storage during sequence
    short mLastHw0;          ///< store last half-word 0
    short mLastHw1;          ///< store last half-word 1
    short mLastHw2;          ///< store last half-word 2
    short mLastHw3;          ///< store last half-word 3
    int mSampa;             ///< SAMPA number
    int mLowHigh;           ///< Low or high bits

};
}
}
#endif
