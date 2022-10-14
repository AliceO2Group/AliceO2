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

/// \file DigitTime.h
/// \brief Definition of the Time Bin container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitTime_H_
#define ALICEO2_TPC_DigitTime_H_

#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TPCSimulation/DigitGlobalPad.h"
#include "SimulationDataFormat/LabelContainer.h"
#include "CommonUtils/DebugStreamer.h"
#include "TPCSimulation/CommonMode.h"

namespace o2::tpc
{

class Digit;

/// \class DigitTime
/// This is the second class of the intermediate Digit Containers, in which all incoming electrons from the hits are
/// sorted into after amplification
/// The structure assures proper sorting of the Digits when later on written out for further processing.
/// This class holds the individual Pad Row containers and is contained within the CRU Container.

class DigitTime
{
 public:
  using Streamer = o2::utils::DebugStreamer;
  using PrevDigitInfoArray = std::array<PrevDigitInfo, Mapper::getPadsInSector()>;

  /// Constructor
  DigitTime();

  /// Destructor
  ~DigitTime() = default;

  /// Resets the container
  void reset();

  /// Get common mode for a given GEM stack
  /// \param gemstack GEM stack of the digit
  /// \return Common mode value in that time bin for a given GEM ROC
  float getCommonMode(const GEMstack& gemstack) const;

  /// Get common mode for a given CRU
  /// \param CRU CRU of the digit
  /// \return Common mode value in that time bin for a given CRU
  float getCommonMode(const CRU& cru) const { return getCommonMode(cru.gemStack()); }

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
  /// \param commonModeOutput Output container for common mode
  /// \param cru CRU ID
  /// \param timeBin Time bin
  /// \param commonMode Common mode value of that specific ROC
  /// \param prevTime Previous time bin to calculate CM and ToT
  template <DigitzationMode MODE>
  void fillOutputContainer(std::vector<Digit>& output, dataformats::MCTruthContainer<MCCompLabel>& mcTruth,
                           std::vector<CommonMode>& commonModeOutput, const Sector& sector, TimeBin timeBin, PrevDigitInfoArray* prevTime = nullptr, Streamer* debugStream = nullptr, const CalPad* itParams[2] = nullptr);

 private:
  std::array<float, GEMSTACKSPERSECTOR> mCommonMode;                 ///< Common mode container - 4 GEM ROCs per sector
  std::array<DigitGlobalPad, Mapper::getPadsInSector()> mGlobalPads; ///< Pad Container for the ADC value
  int mDigitCounter = 0;                                             ///< counts the number of digits in this timebin

  o2::dataformats::LabelContainer<std::pair<MCCompLabel, int>, false> mLabels;
  // std::deque<int> mOccupiedPads; // iterable container of occupied pads
};

inline DigitTime::DigitTime() : mCommonMode(), mGlobalPads()
{
  mCommonMode.fill(0.f);
  mLabels.reserve(Mapper::getPadsInSector() / 3);
}

inline void DigitTime::addDigit(const MCCompLabel& label, const CRU& cru, GlobalPadNumber globalPad, float signal)
{
  auto& paddigit = mGlobalPads[globalPad];
  if (paddigit.getID() == -1) {
    // this means we have a new digit
    paddigit.setID(mDigitCounter++);
    // could also register this pad in a vector of digits
  }

  // previous digit for CM and ToT calculation
  paddigit.addDigit(label, signal, mLabels);
  // mCommonMode[cru.gemStack()] += signal * 0.5; // TODO: Replace 0.5 by k-factor, take into account ion tail
}

inline void DigitTime::reset()
{
  for (auto& pad : mGlobalPads) {
    pad.reset();
  }
  mCommonMode.fill(0.f);
}

inline float DigitTime::getCommonMode(const GEMstack& gemstack) const
{
  /// simple case when there is no external capacitance on the ROC
  const Mapper& mapper = Mapper::instance();
  const auto nPads = mapper.getNumberOfPads(gemstack);
  return mCommonMode[gemstack] / static_cast<float>(nPads);
}

template <DigitzationMode MODE>
inline void DigitTime::fillOutputContainer(std::vector<Digit>& output, dataformats::MCTruthContainer<MCCompLabel>& mcTruth,
                                           std::vector<CommonMode>& commonModeOutput, const Sector& sector, TimeBin timeBin,
                                           PrevDigitInfoArray* prevTime, Streamer* debugStream, const CalPad* itParams[2])
{
  const auto& mapper = Mapper::instance();
  const auto& eleParam = ParameterElectronics::Instance();

  // at this point we only have the pure signals from tracks
  // loop over all pads to calculated ion tail, common mode and ToT for saturated signals
  for (size_t iPad = 0; iPad < mGlobalPads.size(); ++iPad) {
    auto& digit = mGlobalPads[iPad];
    if (prevTime) {
      auto& prevDigit = (*prevTime)[iPad];
      if (prevDigit.hasSignal()) {
        digit.foldSignal(prevDigit, sector.getSector(), iPad, timeBin, debugStream, itParams);
      }
      prevDigit.signal = digit.getChargePad(); // to make hasSignal() check work in next time bin
    }
    const CRU cru = mapper.getCRU(sector, iPad);
    mCommonMode[cru.gemStack()] += digit.getChargePad() * 0.5; // TODO: Replace 0.5 by k-factor, take into account ion tail
  }

  // fill common mode output container
  for (size_t i = 0; i < mCommonMode.size(); ++i) {
    const float cm = getCommonMode(GEMstack(i));
    if (cm > 0.) {
      commonModeOutput.push_back({cm, timeBin, static_cast<unsigned char>(i)});
    }
  }

  for (size_t iPad = 0; iPad < mGlobalPads.size(); ++iPad) {
    auto& digit = mGlobalPads[iPad];
    if (eleParam.doNoiseEmptyPads || (digit.getChargePad() > 0.f)) {
      PrevDigitInfo prevDigit;
      if (prevTime) {
        prevDigit = (*prevTime)[iPad];
      }
      const CRU cru = mapper.getCRU(sector, iPad);
      digit.fillOutputContainer<MODE>(output, mcTruth, cru, timeBin, iPad, mLabels, getCommonMode(cru), prevDigit, debugStream);
    }
  }
}
} // namespace o2::tpc

#endif // ALICEO2_TPC_DigitTime_H_
