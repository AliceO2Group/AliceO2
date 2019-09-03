// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "TPCBase/CRU.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCSimulation/DigitTime.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/CDBInterface.h"

namespace o2
{
namespace tpc
{

class Digit;
class DigitMCMetaData;

/// \class DigitContainer
/// This is the base class of the intermediate Digit Containers, in which all incoming electrons from the hits are
/// sorted into after amplification
/// The structure assures proper sorting of the Digits when later on written out for further processing.
/// This class holds the CRU containers.

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
  /// \param sector Sector to be processed
  /// \param eventTime time stamp of the event
  /// \param isContinuous Switch for continuous readout
  /// \param finalFlush Flag whether the whole container is dumped
  void fillOutputContainer(std::vector<Digit>& output, dataformats::MCTruthContainer<MCCompLabel>& mcTruth, const Sector& sector, TimeBin eventTimeBin = 0, bool isContinuous = true, bool finalFlush = false);

  /// Get the size of the container for one event
  size_t size() const { return mTimeBins.size(); }

 private:
  TimeBin mFirstTimeBin = 0;            ///< First time bin to consider
  TimeBin mEffectiveTimeBin = 0;        ///< Effective time bin of that digit
  TimeBin mTmaxTriggered = 0;           ///< Maximum time bin in case of triggered mode (hard cut at average drift speed with additional margin)
  TimeBin mOffset = 700;                ///< Size of the container for one event
  std::deque<DigitTime> mTimeBins{700}; ///< Time bin Container for the ADC value
};

inline DigitContainer::DigitContainer()
{
  auto& detParam = ParameterDetector::Instance();
  mTmaxTriggered = detParam.TmaxTriggered;
}

inline void DigitContainer::reset()
{
  mFirstTimeBin = 0;
  mEffectiveTimeBin = 0;
  for (auto& time : mTimeBins) {
    time.reset();
  }
}

inline void DigitContainer::reserve(TimeBin eventTimeBin)
{
  if (mTimeBins.size() < mOffset + eventTimeBin - mFirstTimeBin) {
    mTimeBins.resize(mOffset + eventTimeBin - mFirstTimeBin);
  }
}

inline void DigitContainer::addDigit(const MCCompLabel& label, const CRU& cru, TimeBin timeBin, GlobalPadNumber globalPad,
                                     float signal)
{
  mEffectiveTimeBin = timeBin - mFirstTimeBin;
  mTimeBins[mEffectiveTimeBin].addDigit(label, cru, globalPad, signal);
}

} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_DigitContainer_H_
