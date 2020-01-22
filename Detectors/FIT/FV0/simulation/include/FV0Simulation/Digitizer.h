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

#include "FV0Simulation/MCLabel.h"
#include "FV0Simulation/DigitizationParameters.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/BCData.h"
#include "FV0Simulation/Detector.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "MathUtils/CachingTF1.h"
#include "MathUtils/RandomRing.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <array>
#include <vector>
#include <TH1F.h>

namespace o2
{
namespace fv0
{
class Digitizer
{
 public:
  void clear();
  void init();

  void setTimeStamp(long t) { mTimeStamp = t; }
  void setEventId(Int_t id) { mEventId = id; }
  void setSrcId(Int_t id) { mSrcId = id; }
  void setInteractionRecord(const o2::InteractionTimeRecord& ir) { mIntRecord = ir; }

  void process(const std::vector<o2::fv0::Hit>& hits,
               std::vector<o2::fv0::BCData>& digitsBC,
               std::vector<o2::fv0::ChannelData>& digitsCh,
               o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>& labels);

  //  const o2::InteractionRecord& getInteractionRecord() const { return mIntRecord; }
  //  o2::InteractionRecord& getInteractionRecord(o2::InteractionRecord& src) { return mIntRecord; }
  //  uint32_t getOrbit() const { return mIntRecord.orbit; }
  //  uint16_t getBC() const { return mIntRecord.bc; }

 private:
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc) -> InteractionTimeRecord
  Int_t mEventId;                   // ID of the current event
  Int_t mSrcId;                     // signal, background or QED
  std::vector<o2::fv0::MCLabel> mMCLabels;

  long mTimeStamp = 0;                                                                  // TF (run) timestamp
  std::array<std::vector<Float_t>, DigitizationParameters::NCHANNELS> mPmtChargeVsTime; // Charge time series aka analogue signal pulse from PM
  UInt_t mNBins;                                                                        // Number of bins in pulse series
  Float_t mBinSize;                                                                     // Time width of the pulse bin - HPTDC resolution

  // Internal helper methods related to conversion of energy-deposition into photons -> photoelectrons -> el. signal
  Int_t SimulateLightYield(Int_t pmt, Int_t nPhot);
  Float_t SimulateTimeCfd(Int_t channel);
  static Float_t PmtResponse(Float_t x);
  Double_t SinglePhESpectrum(Double_t* x, Double_t* par);
  Float_t mPmtTimeIntegral; //

  // Random rings (optimization; accessing prefilled arrays is 3-4 times faster than GetRandom())
  Int_t mIteratorScintDelay;
  std::array<Float_t, DigitizationParameters::HIT_RANDOM_RING_SIZE> mScintillatorDelay;
  Int_t mIteratorGainVar;
  std::array<Float_t, DigitizationParameters::PHE_RANDOM_RING_SIZE> mGainVar;
  Int_t mIteratorSignalShape;
  std::array<Float_t, DigitizationParameters::PHE_RANDOM_RING_SIZE> mSignalShape;

  ClassDefNV(Digitizer, 1);
};

} // namespace fv0
} // namespace o2

#endif
