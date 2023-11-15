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

#ifndef ALICEO2_TOF_WINDOWFILLER_H_
#define ALICEO2_TOF_WINDOWFILLER_H_

#include "TOFBase/Geo.h"
#include "TOFBase/Digit.h"
#include "TOFBase/Strip.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsTOF/Diagnostic.h"
#include "TOFBase/Utils.h"

namespace o2
{
namespace tof
{

class WindowFiller
{
 public:
  struct PatternData {
    uint32_t pattern;
    int icrate;

    unsigned long row;

    PatternData(uint32_t patt = 0, int icr = 0, unsigned long rw = 0) : pattern(patt), icrate(icr), row(rw) {}
  };

  struct CrateHeaderData {
    int32_t bc[Geo::kNCrate] = {-1};
    uint32_t eventCounter[Geo::kNCrate] = {0};
    CrateHeaderData() { memset(bc, -1, Geo::kNCrate * 4); }
  };

  WindowFiller() { initObj(); };
  ~WindowFiller() = default;

  void initObj();

  void reset();

  uint64_t getCurrentReadoutWindow() const { return mReadoutWindowCurrent; }
  void setCurrentReadoutWindow(uint64_t value) { mReadoutWindowCurrent = value; }
  void setEventTime(InteractionTimeRecord value)
  {
    mEventTime = value;
  }

  std::vector<Digit>* getDigitPerTimeFrame() { return &mDigitsPerTimeFrame; }
  std::vector<ReadoutWindowData>* getReadoutWindowData() { return &mReadoutWindowData; }
  std::vector<ReadoutWindowData>* getReadoutWindowDataFiltered() { return &mReadoutWindowDataFiltered; }
  DigitHeader& getDigitHeader() { return mDigitHeader; }

  template <typename VROF, typename VPAT>
  void setReadoutWindowData(const VROF& row, const VPAT& pattern)
  {
    // copy rowdata info needed to call fillDiagonsticFrequency when reading frm file (digits/ctf). Not needed when digitizing or decoding
    mPatterns.clear();
    mReadoutWindowData.clear();
    for (const auto crow : row) {
      mReadoutWindowData.push_back(crow);
    }

    for (const auto dia : pattern) {
      mPatterns.push_back(dia);
    }
  }

  void setNOrbitInTF(uint32_t norb) { o2::tof::Utils::setNOrbitInTF(norb); }
  void fillOutputContainer(std::vector<Digit>& digits);
  void flushOutputContainer(std::vector<Digit>& digits); // flush all residual buffered data
  void setContinuous(bool value = true) { mContinuous = value; }
  bool isContinuous() const { return mContinuous; }

  void fillDiagnosticFrequency();

  void resizeVectorFutureDigit(int size) { mFutureDigits.resize(size); }

  void setFirstIR(const o2::InteractionRecord& ir) { mFirstIR = ir; }

  void maskNoiseRate(int val) { mMaskNoiseRate = val; }

  void clearCounts()
  {
    memset(mChannelCounts, 0, o2::tof::Geo::NCHANNELS * sizeof(mChannelCounts[0]));
  }

  std::vector<uint8_t>& getPatterns() { return mPatterns; }
  void addPattern(const uint32_t val, int icrate, int orbit, int bc) { mCratePatterns.emplace_back(val, icrate, orbit * 3 + (bc + 100) / Geo::BC_IN_WINDOW); }
  void addCrateHeaderData(unsigned long orbit, int crate, int32_t bc, uint32_t eventCounter);
  Diagnostic& getDiagnosticFrequency() { return mDiagnosticFrequency; }

  void addCount(int channel) { mChannelCounts[channel]++; }

 protected:
  // info TOF timewindow
  uint64_t mReadoutWindowCurrent = 0;
  InteractionRecord mFirstIR{0, 0}; // reference IR (1st IR of the timeframe)
  InteractionTimeRecord mEventTime;

  bool mContinuous = true;
  bool mFutureToBeSorted = false;

  // only needed from Decoder
  int mMaskNoiseRate = -11;
  int mChannelCounts[o2::tof::Geo::NCHANNELS]; // count of channel hits in the current TF (if MaskNoiseRate enabled)

  // digit info
  //std::vector<Digit>* mDigits;

  static const int MAXWINDOWS = 2; // how many readout windows we can buffer

  std::vector<Digit> mDigitsPerTimeFrame;
  std::vector<ReadoutWindowData> mReadoutWindowData;
  std::vector<ReadoutWindowData> mReadoutWindowDataFiltered;

  int mIcurrentReadoutWindow = 0;

  // array of strips to store the digits per strip (one for the current readout window, one for the next one)
  std::vector<Strip> mStrips[MAXWINDOWS];
  std::vector<Strip>* mStripsCurrent = &(mStrips[0]);
  std::vector<Strip>* mStripsNext[MAXWINDOWS - 1];

  // arrays with digit and MCLabels out of the current readout windows (stored to fill future readout window)
  std::vector<Digit> mFutureDigits;

  std::vector<uint8_t> mPatterns;
  std::vector<uint64_t> mErrors;

  Diagnostic mDiagnosticFrequency;

  std::vector<PatternData> mCratePatterns;
  std::vector<CrateHeaderData> mCrateHeaderData;

  DigitHeader mDigitHeader;

  void fillDigitsInStrip(std::vector<Strip>* strips, int channel, int tdc, int tot, uint64_t nbc, UInt_t istrip, uint32_t triggerorbit = 0, uint16_t triggerbunch = 0);
  //  void fillDigitsInStrip(std::vector<Strip>* strips, o2::dataformats::MCTruthContainer<o2::tof::MCLabel>* mcTruthContainer, int channel, int tdc, int tot, int nbc, UInt_t istrip, Int_t trackID, Int_t eventID, Int_t sourceID);

  void checkIfReuseFutureDigits();
  void checkIfReuseFutureDigitsRO();

  void insertDigitInFuture(Int_t channel, Int_t tdc, Int_t tot, uint64_t bc, Int_t label = 0, uint32_t triggerorbit = 0, uint16_t triggerbunch = 0)
  {
    mFutureDigits.emplace_back(channel, tdc, tot, bc, label, triggerorbit, triggerbunch);
    mFutureToBeSorted = true;
  }

  bool isMergable(Digit digit1, Digit digit2)
  {
    if (digit1.getChannel() != digit2.getChannel()) {
      return false;
    }

    if (digit1.getBC() != digit2.getBC()) {
      return false;
    }

    // Check if the difference is larger than the TDC dead time
    if (std::abs(digit1.getTDC() - digit2.getTDC()) > Geo::DEADTIMETDC) {
      return false;
    }
    return true;
  }

  ClassDefNV(WindowFiller, 2);
};
} // namespace tof
} // namespace o2
#endif
