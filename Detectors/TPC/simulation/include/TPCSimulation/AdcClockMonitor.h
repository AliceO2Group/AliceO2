/// \file AdcClockMonitor.h
/// \brief Class to monitor the ADC smapling clock contained in the GBT frame
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_ADCCLOCKMONITOR_H_
#define ALICEO2_TPC_ADCCLOCKMONITOR_H_

#include "FairLogger.h"
#include <iostream>
#include <iomanip>

namespace AliceO2 {
namespace TPC {

/// \class AdcClockMonitor
/// \brief Class to monitor the ADC smapling clock contained in the GBT frame

class AdcClockMonitor {
  public :

    /// Default Constructor
    AdcClockMonitor();

    /// Constructor
    AdcClockMonitor(int sampa);

    /// Copy Constructor
    AdcClockMonitor(const AdcClockMonitor& other);

    /// Destructor
    ~AdcClockMonitor();

    /// Reset function to clear history
    void reset();

    /// Checks whether new sequence is valid
    /// @param seq New sequence of 4 bits
    /// @return Returns outcome of the check, 1 for an error, 0 otherwise
    int addSequence(char seq);

    /// Get the state of the ADC monitor
    /// @return Returns outcome of monitor, 1 for an error, 0 otherwise
    int getState() { return ((mState == state::error) || (mSequenceCompleted == false)) ? 1 : 0 ; };

  private:
    enum state{locked = 0, error = 1};

    int mSampa;                 ///< Store SAMPA ID
    char mPrevSequence;         ///< Store previous 4 bits
    char mTransition0;          ///< 1st transition to look for
    char mTransition1;          ///< 2nd transition to look for
    bool mSequenceCompleted;    ///< Store whether sequence was completed at least once
    int mSequencePosition;      ///< current position in sequence
    state mState;               ///< current state
};
}
}
#endif
