// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FDD_DIGIT_H
#define ALICEO2_FDD_DIGIT_H

#include "CommonDataFormat/RangeReference.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsFDD/ChannelData.h"
#include <Framework/Logger.h>
#include <iosfwd>
#include <Rtypes.h>
#include <gsl/span>
#include <bitset>
#include <tuple>
namespace o2
{
namespace fdd
{

class ChannelData;

struct Triggers {
  enum { bitA,
         bitC,
         bitVertex,
         bitCen,
         bitSCen };
  uint8_t triggersignals = 0; // FDD trigger signals
  int8_t nChanA = 0;          // number of fired channels A side
  int8_t nChanC = 0;          // number of fired channels A side
  int32_t amplA = -1024;      // sum amplitude A side
  int32_t amplC = -1024;      // sum amplitude C side
  int16_t timeA = 0;          // average time A side
  int16_t timeC = 0;          // average time C side
  Triggers() = default;
  Triggers(uint8_t signals, int8_t chanA, int8_t chanC, int32_t aamplA, int32_t aamplC, int16_t atimeA, int16_t atimeC)
  {
    triggersignals = signals;
    nChanA = chanA;
    nChanC = chanC;
    amplA = aamplA;
    amplC = aamplC;
    timeA = atimeA;
    timeC = atimeC;
  }

  bool getOrA() const { return (triggersignals & (1 << bitA)) != 0; }
  bool getOrC() const { return (triggersignals & (1 << bitC)) != 0; }
  bool getVertex() const { return (triggersignals & (1 << bitVertex)) != 0; }
  bool getCen() const { return (triggersignals & (1 << bitCen)) != 0; }
  bool getSCen() const { return (triggersignals & (1 << bitSCen)) != 0; }

  void cleanTriggers()
  {
    triggersignals = 0;
    nChanA = nChanC = 0;
    amplA = amplC = 0;
    timeA = timeC = 0;
  }
  Triggers getTriggers();
  bool operator==(Triggers const& other) const
  {
    return std::tie(triggersignals, nChanA, nChanC, amplA, amplC, timeA, timeC) ==
           std::tie(other.triggersignals, other.nChanA, other.nChanC, other.amplA, other.amplC, other.timeA, other.timeC);
  }
  void printLog() const
  {
    LOG(INFO) << "mTrigger: " << static_cast<uint16_t>(triggersignals);
    LOG(INFO) << "nChanA: " << static_cast<uint16_t>(nChanA) << " | nChanC: " << static_cast<uint16_t>(nChanC);
    LOG(INFO) << "amplA: " << amplA << " | amplC: " << amplC;
    LOG(INFO) << "timeA: " << timeA << " | timeC: " << timeC;
  }
  ClassDefNV(Triggers, 1);
};

struct DetTrigInput {
  static constexpr char sChannelNameDPL[] = "TRIGGERINPUT";
  static constexpr char sDigitName[] = "DetTrigInput";
  static constexpr char sDigitBranchName[] = "FDDTRIGGERINPUT";
  o2::InteractionRecord mIntRecord; // bc/orbit of the intpur
  std::bitset<5> mInputs;           // pattern of inputs.
  DetTrigInput() = default;
  DetTrigInput(const o2::InteractionRecord& iRec, Bool_t isA, Bool_t isC, Bool_t isVrtx, Bool_t isCnt, Bool_t isSCnt)
    : mIntRecord(iRec),
      mInputs((isA << Triggers::bitA) |
              (isC << Triggers::bitC) |
              (isVrtx << Triggers::bitVertex) |
              (isCnt << Triggers::bitCen) |
              (isSCnt << Triggers::bitSCen))
  {
  }
  ClassDefNV(DetTrigInput, 1);
};

struct Digit {
  static constexpr char sChannelNameDPL[] = "DIGITSBC";
  static constexpr char sDigitName[] = "Digit";
  static constexpr char sDigitBranchName[] = "FDDDigit";
  o2::dataformats::RangeRefComp<5> ref;
  Triggers mTriggers;               // pattern of triggers  in this BC
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)
  Digit() = default;
  Digit(int first, int ne, o2::InteractionRecord iRec, Triggers chTrig) : ref(first, ne), mIntRecord(iRec), mTriggers(chTrig) {}
  typedef DetTrigInput DetTrigInput_t;
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }
  o2::InteractionRecord getIntRecord() const { return mIntRecord; };
  void setIntRecord(const o2::InteractionRecord& intRec) { mIntRecord = intRec; }
  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const
  {
    return ref.getEntries() ? gsl::span<const ChannelData>(&tfdata[ref.getFirstEntry()], ref.getEntries()) : gsl::span<const ChannelData>();
  }
  DetTrigInput makeTrgInput() const { return DetTrigInput{mIntRecord, mTriggers.getOrA(), mTriggers.getOrC(), mTriggers.getVertex(), mTriggers.getCen(), mTriggers.getSCen()}; }
  void fillTrgInputVec(std::vector<DetTrigInput>& vecTrgInput) const
  {
    vecTrgInput.emplace_back(mIntRecord, mTriggers.getOrA(), mTriggers.getOrC(), mTriggers.getVertex(), mTriggers.getCen(), mTriggers.getSCen());
  }
  bool operator==(const Digit& other) const
  {
    return std::tie(ref, mTriggers, mIntRecord) == std::tie(other.ref, other.mTriggers, other.mIntRecord);
  }
  void printLog() const
  {
    LOG(INFO) << "______________DIGIT DATA____________";
    LOG(INFO) << "BC: " << mIntRecord.bc << "| ORBIT: " << mIntRecord.orbit;
    LOG(INFO) << "Ref first: " << ref.getFirstEntry() << "| Ref entries: " << ref.getEntries();
    mTriggers.printLog();
  }
  ClassDefNV(Digit, 3);
};
//For TCM extended mode (calibration mode), TCMdataExtended digit
struct TriggersExt {
  static constexpr char sChannelNameDPL[] = "DIGITSTRGEXT";
  static constexpr char sDigitName[] = "TriggersExt";
  static constexpr char sDigitBranchName[] = "FDDDigitTrgExt";
  TriggersExt(std::array<uint32_t, 20> triggerWords) : mTriggerWords(triggerWords) {}
  TriggersExt() = default;
  o2::InteractionRecord mIntRecord;
  void setTrgWord(uint32_t trgWord, std::size_t pos) { mTriggerWords[pos] = trgWord; }
  std::array<uint32_t, 20> mTriggerWords;
  void printLog() const
  {
    LOG(INFO) << "______________EXTENDED TRIGGERS____________";
    LOG(INFO) << "BC: " << mIntRecord.bc << "| ORBIT: " << mIntRecord.orbit;
    for (int i = 0; i < 20; i++) {
      LOG(INFO) << "N: " << i + 1 << " | TRG: " << mTriggerWords[i];
    }
  }
  ClassDefNV(TriggersExt, 1);
};

} // namespace fdd
} // namespace o2
#endif
