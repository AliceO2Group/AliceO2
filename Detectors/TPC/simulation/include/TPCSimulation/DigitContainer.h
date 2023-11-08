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

/// \file DigitContainer.h
/// \brief Definition of the Digit Container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitContainer_H_
#define ALICEO2_TPC_DigitContainer_H_

#include <deque>
#include <algorithm>
#include "TPCBase/CRU.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCSimulation/DigitTime.h"
#include "CommonUtils/DebugStreamer.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"

namespace o2::tpc
{

class Digit;

/// \class DigitContainer
/// This is the base class of the intermediate Digit Containers, in which all incoming electrons from the hits are
/// sorted into after amplification
/// The structure assures proper sorting of the Digits when later on written out for further processing.
/// This class holds the time bin containers.

class DigitContainer
{
 public:
  /// Default constructor
  DigitContainer();

  /// Destructor
  ~DigitContainer() = default;

  /// Reset the container
  void reset();

  /// Reserve space in the container for a given event
  void reserve(TimeBin eventTimeBin);

  /// Set the start time of the first event
  /// \param time Time of the first event
  void setStartTime(TimeBin time) { mFirstTimeBin = time; }

  /// Add digit to the container
  /// \param eventID MC Event ID
  /// \param trackID MC Track ID
  /// \param cru CRU of the digit
  /// \param globalPad Global pad number of the digit
  /// \param timeBin Time bin of the digit
  /// \param signal Charge of the digit in ADC counts
  void addDigit(const MCCompLabel& label, const CRU& cru, TimeBin timeBin, GlobalPadNumber globalPad, float signal);

  /// Fill output vector
  /// \param output Output container
  /// \param mcTruth MC Truth container
  /// \param commonModeOutput Output container for the common mode
  /// \param sector Sector to be processed
  /// \param eventTime time stamp of the event
  /// \param isContinuous Switch for continuous readout
  /// \param finalFlush Flag whether the whole container is dumped
  void fillOutputContainer(std::vector<Digit>& output, dataformats::MCTruthContainer<MCCompLabel>& mcTruth, std::vector<CommonMode>& commonModeOutput, const Sector& sector, TimeBin eventTimeBin = 0, bool isContinuous = true, bool finalFlush = false);

  /// Get the size of the container for one event
  size_t size() const { return mTimeBins.size(); }

 private:
  TimeBin mFirstTimeBin = 0;                                  ///< First time bin to consider
  TimeBin mEffectiveTimeBin = 0;                              ///< Effective time bin of that digit
  TimeBin mTmaxTriggered = 0;                                 ///< Maximum time bin in case of triggered mode (hard cut at average drift speed with additional margin)
  TimeBin mOffset;                                            ///< Size of the container for one event
  std::deque<DigitTime*> mTimeBins;                           ///< Time bin Container for the ADC value
  std::unique_ptr<DigitTime::PrevDigitInfoArray> mPrevDigArr; ///< Keep track of ToT and ion tail cumul from last time bin
  o2::utils::DebugStreamer mStreamer;                         ///< Debug streamer

  void reportSettings();
};

inline DigitContainer::DigitContainer()
{
  auto& detParam = ParameterDetector::Instance();
  auto& gasParam = ParameterGas::Instance();
  auto& eleParam = ParameterElectronics::Instance();
  mTmaxTriggered = detParam.TmaxTriggered;

  // always have 50 % contingency for the size of the container depending on the input
  mOffset = static_cast<TimeBin>(detParam.TPCRecoWindowSim * detParam.TPClength / gasParam.DriftV / eleParam.ZbinWidth);
  mTimeBins.resize(mOffset, nullptr);
}

inline void DigitContainer::reset()
{
  mFirstTimeBin = 0;
  mEffectiveTimeBin = 0;
  for (auto& time : mTimeBins) {
    if (time) {
      time->reset();
    }
  }
  if (mPrevDigArr) {
    std::fill(mPrevDigArr->begin(), mPrevDigArr->end(), PrevDigitInfo{});
  }
}

inline void DigitContainer::reserve(TimeBin eventTimeBin)
{
  const auto space = mOffset + eventTimeBin - mFirstTimeBin;
  if (mTimeBins.size() < space) {
    mTimeBins.resize(space);
  }
}

inline void DigitContainer::addDigit(const MCCompLabel& label, const CRU& cru, TimeBin timeBin, GlobalPadNumber globalPad,
                                     float signal)
{
  mEffectiveTimeBin = timeBin - mFirstTimeBin;
  if (mEffectiveTimeBin >= mTimeBins.size()) {
    // LOG(warning) << "Out of bound access to digit container .. dropping digit";
    return;
  }

  if (mTimeBins[mEffectiveTimeBin] == nullptr) {
    mTimeBins[mEffectiveTimeBin] = new DigitTime();
  }

  mTimeBins[mEffectiveTimeBin]->addDigit(label, cru, globalPad, signal);
}

} // namespace o2::tpc

#endif // ALICEO2_TPC_DigitContainer_H_
