/// \file SyncPatternMonitor.h
/// \brief Class to monitor the data stream and detect synchronization patterns
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_SYNCPATTERNMONITOR_H_
#define ALICEO2_TPC_SYNCPATTERNMONITOR_H_

#include "FairLogger.h"
#include <iostream>
#include <iomanip>
#include <array>

namespace o2 {
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
    ~SyncPatternMonitor() = default;

    /// Reset function to clear history
    void reset();

    /// Adds a sequence of 4 new half-words and looks for sync pattern
    /// @param hw0 1st (timewise) half word
    /// @param hw1 2nd (timewise) half word
    /// @param hw2 3th (timewise) half word
    /// @param hw3 4th (timewise) half word
    /// @return Position of first part of the synchronization pattern, -1 if no pattern was found
    short addSequence(const short hw0, const short hw1, const short hw2, const short hw3);

    /// Get position
    /// @return Position of first part of the synchronization pattern, -1 if no patter was found
    short getPosition() { return mPatternFound ? mHwWithPattern : -1; };

  private:

    const static short SYNC_START = 2;
    const static short PATTERN_A = 0x15;
    const static short PATTERN_B = 0x0A;
    static constexpr std::array<short,32> SYNC_PATTERN {{
      PATTERN_A, PATTERN_A, PATTERN_B, PATTERN_B, PATTERN_A, PATTERN_A, PATTERN_B, PATTERN_B,
      PATTERN_A, PATTERN_A, PATTERN_B, PATTERN_B, PATTERN_A, PATTERN_A, PATTERN_B, PATTERN_B,
      PATTERN_A, PATTERN_A, PATTERN_A, PATTERN_A, PATTERN_B, PATTERN_B, PATTERN_B, PATTERN_B,
      PATTERN_A, PATTERN_A, PATTERN_A, PATTERN_A, PATTERN_B, PATTERN_B, PATTERN_B, PATTERN_B
    }};

    void patternFound(const short hw); 

    void checkWord(const short hw, const short pos);

    bool mPatternFound;     ///< store whether pattern was already found
    short mPosition;        ///< position of last part of the pattern
    short mHwWithPattern;   ///< Half word which startet with the pattern
    int mSampa;             ///< SAMPA number
    int mLowHigh;           ///< Low or high bits
    unsigned mCheckedWords; ///< Counter for half words got checked

};

inline
short SyncPatternMonitor::addSequence(const short hw0, const short hw1, const short hw2, const short hw3) {
  checkWord(hw0,1); checkWord(hw1,2);
  checkWord(hw2,3); checkWord(hw3,0);
  return getPosition();
};

inline
void SyncPatternMonitor::checkWord(const short hw, const short pos) {
  ++mCheckedWords;
  if (hw == SYNC_PATTERN[mPosition]) ++mPosition;
  else mPosition = SYNC_START;
  if (mPosition == 31) patternFound(pos); 
};

inline
void SyncPatternMonitor::patternFound(const short hw) { 
  LOG(INFO) << "SAMPA " << mSampa << " (" << ((mLowHigh == 0) ? " low" : "high") << "): "
     << "SYNC found at " << hw << " in " << mCheckedWords << FairLogger::endl;
  if (mPatternFound) {
    LOG(WARNING) << "SAMPA " << mSampa << " (" << ((mLowHigh == 0) ? " low" : "high") << "): "
      << "SYNC was already found" << FairLogger::endl;
  }
  mPatternFound = true; 
  mPosition = SYNC_START; 
  mHwWithPattern = hw;
};

}
}
#endif
