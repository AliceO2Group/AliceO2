// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//_____________________________________________
// GENERAL IDEA
//
// - Focus on https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/TPC/simulation/include/TPCSimulation/DigitContainer.h
// - Comform it to the classes used by EMCAL in https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/EMCAL/simulation/include/EMCALSimulation/DigitsWriteoutBuffer.h
//   and in https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/EMCAL/simulation/include/EMCALSimulation/DigitsVectorStream.h
// - The two concepts are similar
// - EMCAL's was however adapted to triggered mode
// - The principle is to reapply the old TPC format to the continuous readout directly
//
// 1) the idea of dumping the time bins in 15 bins is dumped
// 2) the data are pushed to the output stream whenever possible
// 3) the EndOfRun is implemented to break the push to stream
//    it is a flag set to 1 which is read by the filloutputcontainer
// 4) no more check of start or end of trigger
// 5) what about the precollision flags?
// 6) is there any longer a need to set the time of the sampled digits time?

#ifndef ALICEO2_EMCAL_DIGITSWRITEOUTBUFFERTRU_H_
#define ALICEO2_EMCAL_DIGITSWRITEOUTBUFFERTRU_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <deque>
#include <list>
#include <gsl/span>
#include "DataFormatsEMCAL/Digit.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "EMCALSimulation/LZEROElectronics.h"
#include "EMCALSimulation/DigitTimebin.h"

// using namespace o2::emcal;

namespace o2
{
namespace emcal
{

/// \class DigitsWriteoutBufferTRU
/// \brief Container class for time sampled digits to be sent to TRUs in true continuous readout
/// \ingroup EMCALsimulation
/// \author Hadi Hassan, ORNL
/// \author Markus Fasel, ORNL
/// \author Simone Ragoni, ORNL
/// \date 27/09/2022

class DigitsWriteoutBufferTRU
{
 public:
  /// Default constructor
  DigitsWriteoutBufferTRU(unsigned int nTimeBins = 15);

  /// Destructor
  ~DigitsWriteoutBufferTRU() = default;

  /// clear the container
  void clear();

  /// clear DigitsVectorStream
  void flush()
  {
    // mDigitStream.clear();
  }

  void init();

  /// Reserve space for the future container
  /// \param eventTimeBin resize adding at the end
  void reserve(int eventTimeBin);

  /// This is for the readout window that was interrupted by the end of the run
  void finish();

  /// Add digit to the container
  /// \param towerID Cell ID
  /// \param dig Labaled digit to add
  void addDigits(unsigned int towerID, std::vector<o2::emcal::Digit>& digList);

  /// Fill output streamer
  /// \param isEndOfTimeFrame End of Time Frame
  /// \param nextInteractionRecord Next interaction record, to compute the amount of TimeBins to be saved
  void fillOutputContainer(bool isEndOfTimeFrame, InteractionRecord& nextInteractionRecord, std::vector<TRUElectronics>& patchesFromAllTRUs, LZEROElectronics& LZERO);

  /// Setters for the live time, busy time, pre-trigger time
  void setLiveTime(unsigned int liveTime) { mLiveTime = liveTime; }
  void setBusyTime(unsigned int busyTime) { mBusyTime = busyTime; }

  const std::deque<o2::emcal::DigitTimebinTRU>& getTimeBins() const { return mTimeBins; }

 private:
  unsigned int mBufferSize = 15;  ///< The size of the buffer
  unsigned int mLiveTime = 1500;  ///< EMCal live time (ns)
  unsigned int mBusyTime = 35000; ///< EMCal busy time (ns)
  // unsigned int mPreTriggerTime = 600;               ///< EMCal pre-trigger time (ns)
  unsigned long mTriggerTime = 0;                   ///< Time of the collision that fired the trigger (ns)
  unsigned long mLastEventTime = 0;                 ///< The event time of last collisions in the readout window
  unsigned int mPhase = 0;                          ///< The event L1 phase
  bool mFirstEvent = true;                          ///< Flag to the first event in the run
  std::deque<o2::emcal::DigitTimebinTRU> mTimeBins; ///< Container for time sampled digits per tower ID for continuous digits
  unsigned int mFirstTimeBin = 0;
  bool mEndOfRun = 0;
  bool mNoPileupMode = false;                      ///< pileup mode from SimParam
  o2::InteractionRecord mCurrentInteractionRecord; ///< Interaction Record of the current event, to be used to fill the output container

  ClassDefNV(DigitsWriteoutBufferTRU, 5);
  // ClassDefNV are for objects which do not inherit from tobject
  // you do not need classIMP instead
};

} // namespace emcal

} // namespace o2

#endif /* ALICEO2_EMCAL_DIGITSWRITEOUTBUFFERTRU_H_ */