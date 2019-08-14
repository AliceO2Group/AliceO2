// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SyncPatternMonitor.h
/// \brief Class to monitor the data stream and detect synchronization patterns
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_SYNCPATTERNMONITOR_H_
#define ALICEO2_TPC_SYNCPATTERNMONITOR_H_

#include "FairLogger.h"
#include <iosfwd>
#include <iomanip>
#include <array>

namespace o2
{
namespace tpc
{

/// \class SyncPatternMonitor
/// \brief Class to monitor the data stream and detect synchronization patterns

class SyncPatternMonitor
{
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

  /// Return a Pattern sync pattern
  /// @return Pattern A
  short getPatternA() const { return PATTERN_A; };

  /// Return a Pattern sync pattern
  /// @return Pattern B
  short getPatternB() const { return PATTERN_B; };

  // Return first position to look for in whole pattern
  /// @return start
  short getSyncStart() const { return SYNC_START; };

 private:
  const static short SYNC_START = 2;
  const static short PATTERN_A = 0x15;
  const static short PATTERN_B = 0x0A;
  static constexpr std::array<short, 32> SYNC_PATTERN{{PATTERN_A, PATTERN_A, PATTERN_B, PATTERN_B, PATTERN_A, PATTERN_A, PATTERN_B, PATTERN_B,
                                                       PATTERN_A, PATTERN_A, PATTERN_B, PATTERN_B, PATTERN_A, PATTERN_A, PATTERN_B, PATTERN_B,
                                                       PATTERN_A, PATTERN_A, PATTERN_A, PATTERN_A, PATTERN_B, PATTERN_B, PATTERN_B, PATTERN_B,
                                                       PATTERN_A, PATTERN_A, PATTERN_A, PATTERN_A, PATTERN_B, PATTERN_B, PATTERN_B, PATTERN_B}};

  void patternFound(const short hw);

  void checkWord(const short hw, const short pos);

  bool mPatternFound;     ///< store whether pattern was already found
  short mPosition;        ///< position of last part of the pattern
  short mHwWithPattern;   ///< Half word which startet with the pattern
  int mSampa;             ///< SAMPA number
  int mLowHigh;           ///< Low or high bits
  unsigned mCheckedWords; ///< Counter for half words got checked
};

inline short SyncPatternMonitor::addSequence(const short hw0, const short hw1, const short hw2, const short hw3)
{
  checkWord(hw0, 1);
  checkWord(hw1, 2);
  checkWord(hw2, 3);
  checkWord(hw3, 0);
  return mPatternFound; //getPosition();
};

inline void SyncPatternMonitor::checkWord(const short hw, const short pos)
{
  ++mCheckedWords;
  if (hw == SYNC_PATTERN[mPosition])
    ++mPosition;
  else if (!(mPosition == SYNC_START + 2 && hw == SYNC_PATTERN[mPosition - 1]))
    mPosition = SYNC_START;
  // Don't reset mPosition at the beginning to avoid missing of start of sync pattern in cases like
  //
  //
  //       ... A  A  A  B  B ...
  //
  //           ^  ^  ^
  // random A _|  |  |_ would trigger reset
  //              |
  //             real start

  if (mPosition == 32)
    patternFound(pos);
};

inline void SyncPatternMonitor::patternFound(const short hw)
{
  LOG(DEBUG) << "SAMPA " << mSampa << " (" << ((mLowHigh == 0) ? " low" : "high") << "): "
             << "SYNC found at Position " << hw << " in checked half word #" << mCheckedWords;
  if (mPatternFound) {
    LOG(WARNING) << "SAMPA " << mSampa << " (" << ((mLowHigh == 0) ? " low" : "high") << "): "
                 << "SYNC was already found";
  }
  mPatternFound = true;
  mPosition = SYNC_START;
  mHwWithPattern = hw;
};

} // namespace tpc
} // namespace o2
#endif
