// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FV0_DIGITIZER_H
#define ALICEO2_FV0_DIGITIZER_H

#include <FV0Simulation/MCLabel.h>
#include <FV0Simulation/DigitizationParameters.h>
#include <DataFormatsFV0/ChannelData.h>
#include <DataFormatsFV0/BCData.h>
#include <FV0Simulation/Detector.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include <MathUtils/RandomRing.h>
#include <CommonDataFormat/InteractionRecord.h>
#include <array>
#include <vector>

namespace o2
{
namespace fv0
{
class Digitizer
{
 private:
  using DP = DigitizationParameters;
  typedef math_utils::RandomRing<float_v::size() * DP::HIT_RANDOM_RING_SIZE> HitRandomRingType;
  typedef math_utils::RandomRing<float_v::size() * DP::PHE_RANDOM_RING_SIZE> PheRandomRingType;

 public:
  Digitizer()
    : mTimeStamp(0), mIntRecord(), mEventId(-1), mSrcId(-1), mMCLabels(), mPmtChargeVsTime(), mNBins(), mRndScintDelay(HitRandomRingType::RandomType::CustomTF1), mRndGainVar(PheRandomRingType::RandomType::CustomTF1), mRndSignalShape(PheRandomRingType::RandomType::CustomTF1), mPmtResponseTables()
  {
  }

  /// Destructor
  ~Digitizer() = default;

  Digitizer(const Digitizer&) = delete;
  Digitizer& operator=(const Digitizer&) = delete;

  void clear();
  void init();

  void setTimeStamp(long t) { mTimeStamp = t; }
  void setEventId(Int_t id) { mEventId = id; }
  void setSrcId(Int_t id) { mSrcId = id; }
  void setInteractionRecord(const InteractionTimeRecord& ir) { mIntRecord = ir; }

  void process(const std::vector<fv0::Hit>& hits,
               std::vector<fv0::BCData>& digitsBC,
               std::vector<fv0::ChannelData>& digitsCh,
               dataformats::MCTruthContainer<fv0::MCLabel>& labels);

  const InteractionRecord& getInteractionRecord() const { return mIntRecord; }
  InteractionRecord& getInteractionRecord(InteractionRecord& src) { return mIntRecord; }
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }

 private:
  long mTimeStamp;              // TF (run) timestamp
  InteractionRecord mIntRecord; // Interaction record (orbit, bc) -> InteractionTimeRecord
  Int_t mEventId;               // ID of the current event
  Int_t mSrcId;                 // signal, background or QED
  std::vector<fv0::MCLabel> mMCLabels;

  std::array<std::vector<Float_t>, DP::NCHANNELS> mPmtChargeVsTime; // Charge time series aka analogue signal pulse from PM
  UInt_t mNBins;                                                    // Number of bins in pulse series
  Float_t mBinSize;                                                 // Time width of the pulse bin - HPTDC resolution
  Float_t mPmtTimeIntegral;                                         //

  // Random rings
  HitRandomRingType mRndScintDelay;
  PheRandomRingType mRndGainVar;
  PheRandomRingType mRndSignalShape;

  // 8 tables starting at different sub-bin positions, i.e, [-4:4] / 8 * mBinSize
  // wit each table containg values for start + [-2:2:mBinSize] * DigitizationParameters::mPmtTransitTime
  std::array<std::vector<Float_t>, DP::NUM_PMT_RESPONSE_TABLES> mPmtResponseTables;

  // Internal helper methods related to conversion of energy-deposition into photons -> photoelectrons -> el. signal
  Int_t SimulateLightYield(Int_t pmt, Int_t nPhot) const;
  Float_t SimulateTimeCfd(Int_t channel) const;

  static Double_t PmtResponse(Double_t x);
  static Double_t PmtResponse(Double_t* x, Double_t*);
  static Double_t SinglePhESpectrum(Double_t* x, Double_t* par);

  ClassDefNV(Digitizer, 1);
};

} // namespace fv0
} // namespace o2

#endif
