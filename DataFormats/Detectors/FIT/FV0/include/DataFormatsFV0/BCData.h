// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _FV0_BC_DATA_H_
#define _FV0_BC_DATA_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include <Rtypes.h>
#include <gsl/span>
#include <bitset>
#include <vector>
#include <tuple>
/// \file BCData.h
/// \brief Class to describe fired triggered and/or stored channels for the BC and to refer to channel data
/// \author ruben.shahoyan@cern.ch -> maciej.slupecki@cern.ch

namespace o2
{
namespace fv0
{
class ChannelData;

struct Triggers {
  enum {
    bitMinBias,
    bitMinBiasInner, // experimental
    bitMinBiasOuter, // experimental
    bitHighMult,
    bitDummy // non-defined yet, placeholder
  };
  uint8_t triggerSignals = 0; // V0 trigger signals
  int8_t nChanA = 0;          // number of fired channels [A side]
  int32_t amplA = -1000;      // sum amplitude [A side]
  Triggers() = default;
  Triggers(uint8_t signals, int8_t chanA, int32_t amplASum)
  {
    triggerSignals = signals;
    nChanA = chanA;
    amplA = amplASum;
  }

  bool getIsMinBias() const { return (triggerSignals & (1 << bitMinBias)) != 0; }
  bool getIsMinBiasInner() const { return (triggerSignals & (1 << bitMinBiasInner)) != 0; }
  bool getIsMinBiasOuter() const { return (triggerSignals & (1 << bitMinBiasOuter)) != 0; }
  bool getIsHighMult() const { return (triggerSignals & (1 << bitHighMult)) != 0; }
  bool getIsDummy() const { return (triggerSignals & (1 << bitDummy)) != 0; }
  void setTriggers(Bool_t isMinBias, Bool_t isMinBiasInner, Bool_t isMinBiasOuter, Bool_t isHighMult, Bool_t isDummy, int8_t chanA, int32_t amplASum)
  {
    triggerSignals = (isMinBias << bitMinBias) | (isMinBiasInner << bitMinBiasInner) | (isMinBiasOuter << bitMinBiasOuter) | (isHighMult << bitHighMult) | (isDummy << bitDummy);
    nChanA = chanA;
    amplA = amplASum;
  }
  bool operator==(Triggers const& other) const
  {
    //Will be implemented later
    //return std::tie(triggersignals, nChanA, nChanC, amplA, amplC, timeA, timeC) ==
    //       std::tie(other.triggersignals, other.nChanA, other.nChanC, other.amplA, other.amplC, other.timeA, other.timeC);
    return std::tie(triggerSignals, nChanA, amplA) ==
           std::tie(other.triggerSignals, other.nChanA, other.amplA);
  }
  void printLog() const;
  ClassDefNV(Triggers, 1);
};

struct DetTrigInput {
  static constexpr char sChannelNameDPL[] = "TRIGGERINPUT";
  static constexpr char sDigitName[] = "DetTrigInput";
  static constexpr char sDigitBranchName[] = "FV0TRIGGERINPUT";
  o2::InteractionRecord mIntRecord; // bc/orbit of the intpur
  std::bitset<5> mInputs;           // pattern of inputs.
  DetTrigInput() = default;
  DetTrigInput(const o2::InteractionRecord& iRec, Bool_t isMb, Bool_t isMbIn, Bool_t isMbOut, Bool_t isHm, Bool_t isDummy)
    : mIntRecord(iRec),
      mInputs((isMb << Triggers::bitMinBias) |
              (isMbIn << Triggers::bitMinBiasInner) |
              (isMbOut << Triggers::bitMinBiasOuter) |
              (isHm << Triggers::bitHighMult) |
              (isDummy << Triggers::bitDummy))
  {
  }
  ClassDefNV(DetTrigInput, 1);
};

struct BCData {
  static constexpr char sChannelNameDPL[] = "DIGITSBC";
  static constexpr char sDigitName[] = "BCData";
  static constexpr char sDigitBranchName[] = "FV0DigitBC";
  /// we are going to refer to at most 48 channels, so 6 bits for the number of channels and 26 for the reference
  o2::dataformats::RangeRefComp<6> ref;
  o2::InteractionRecord ir; //FV0 is detected by using this field!!!
  Triggers mTriggers;
  BCData() = default;
  BCData(int first, int ne, o2::InteractionRecord iRec, const Triggers& chTrig)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
    ir = iRec;
    mTriggers = chTrig;
  }
  typedef DetTrigInput DetTrigInput_t;
  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const;
  const o2::InteractionRecord& getIntRecord() const { return ir; };
  Triggers getTriggers() const { return mTriggers; }
  void setIntRecord(const o2::InteractionRecord& intRec) { ir = intRec; }
  void setTriggers(Triggers triggers) { mTriggers = triggers; };
  void print() const;
  bool operator==(const BCData& other) const
  {
    return std::tie(ref, mTriggers, ir) == std::tie(other.ref, other.mTriggers, other.ir);
  }
  void printLog() const;
  DetTrigInput makeTrgInput() const { return DetTrigInput{ir, mTriggers.getIsMinBias(), mTriggers.getIsMinBiasInner(), mTriggers.getIsMinBiasOuter(), mTriggers.getIsHighMult(), mTriggers.getIsDummy()}; }
  void fillTrgInputVec(std::vector<DetTrigInput>& vecTrgInput) const
  {
    vecTrgInput.emplace_back(ir, mTriggers.getIsMinBias(), mTriggers.getIsMinBiasInner(), mTriggers.getIsMinBiasOuter(), mTriggers.getIsHighMult(), mTriggers.getIsDummy());
  }
  ClassDefNV(BCData, 1);
};

//For TCM extended mode (calibration mode), TCMdataExtended digit
struct TriggersExt {
  TriggersExt(std::array<uint32_t, 20> triggerWords) : mTriggerWords(triggerWords) {}
  TriggersExt() = default;
  static constexpr char sChannelNameDPL[] = "DIGITSTRGEXT";
  static constexpr char sDigitName[] = "TriggersExt";
  static constexpr char sDigitBranchName[] = "FV0DIGITSTRGEXT";
  o2::InteractionRecord mIntRecord;
  void setTrgWord(uint32_t trgWord, std::size_t pos) { mTriggerWords[pos] = trgWord; }
  std::array<uint32_t, 20> mTriggerWords;
  void printLog() const;
  ClassDefNV(TriggersExt, 1);
};
} // namespace fv0
} // namespace o2

#endif
