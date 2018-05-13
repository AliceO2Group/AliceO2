// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitTime.h
/// \brief Definition of the Time Bin container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitTime_H_
#define ALICEO2_TPC_DigitTime_H_

#include "TPCBase/Mapper.h"
#include "TPCSimulation/DigitGlobalPad.h"

namespace o2
{
namespace TPC
{

class Digit;
class DigitMCMetaData;

/// \class DigitTime
/// This is the third class of the intermediate Digit Containers, in which all incoming electrons from the hits are
/// sorted into after amplification
/// The structure assures proper sorting of the Digits when later on written out for further processing.
/// This class holds the individual Pad Row containers and is contained within the CRU Container.

class DigitTime
{
 public:
  /// Constructor
  DigitTime();

  /// Destructor
  ~DigitTime() = default;

  /// Resets the container
  void reset();

  /// Get common mode for a given GEM stack
  /// \param CRU CRU of the digit
  /// \return Common mode value in that time bin for a given GEM ROC
  float getCommonMode(const CRU& cru) const;

  /// Add digit to the row container
  /// \param eventID MC Event ID
  /// \param trackID MC Track ID
  /// \param cru CRU of the digit
  /// \param globalPad Global pad number of the digit
  /// \param signal Charge of the digit in ADC counts
  void addDigit(const MCCompLabel& label, const CRU& cru, GlobalPadNumber globalPad, float signal);

  /// Fill output vector
  /// \param output Output container
  /// \param mcTruth MC Truth container
  /// \param debug Optional debug output container
  /// \param cru CRU ID
  /// \param timeBin Time bin
  /// \param commonMode Common mode value of that specific ROC
  void fillOutputContainer(std::vector<Digit>* output, dataformats::MCTruthContainer<MCCompLabel>& mcTruth,
                           std::vector<DigitMCMetaData>* debug, const Sector& sector, TimeBin timeBin,
                           float commonMode = 0.f);

 private:
  std::array<float, GEMSTACKSPERSECTOR> mCommonMode;                 ///< Common mode container - 4 GEM ROCs per sector
  std::array<DigitGlobalPad, Mapper::getPadsInSector()> mGlobalPads; ///< Pad Container for the ADC value
};

inline DigitTime::DigitTime() : mCommonMode(), mGlobalPads() { mCommonMode.fill(0); }

inline void DigitTime::addDigit(const MCCompLabel& label, const CRU& cru, GlobalPadNumber globalPad, float signal)
{
  mGlobalPads[globalPad].addDigit(label, signal);
  mCommonMode[cru.gemStack()] += signal;
}

inline void DigitTime::reset()
{
  for (auto& pad : mGlobalPads) {
    pad.reset();
  }
  mCommonMode.fill(0);
}

inline float DigitTime::getCommonMode(const CRU& cru) const
{
  static const Mapper& mapper = Mapper::instance();
  const auto gemStack = static_cast<int>(cru.gemStack());
  const auto nPads = mapper.getNumberOfPads(gemStack);
  return mCommonMode[gemStack] /
         static_cast<float>(nPads); /// simple case when there is no external capacitance on the ROC;
}
}
}

#endif // ALICEO2_TPC_DigitTime_H_
