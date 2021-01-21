// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FDD_DIGITIZER_H
#define ALICEO2_FDD_DIGITIZER_H

#include "DataFormatsFDD/Hit.h"
#include "DataFormatsFDD/ChannelData.h"
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/MCLabel.h"
#include "FDDSimulation/Detector.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "FDDSimulation/DigitizationParameters.h"
#include "FDDBase/Constants.h"
#include "MathUtils/RandomRing.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonConstants/LHCConstants.h"
#include <vector>
#include <array>
#include <deque>
#include <bitset>

namespace o2
{
namespace fdd
{
class Digitizer
{

 private:
  typedef math_utils::RandomRing<float_v::size() * DigitizationParameters::PheRRSize> HitRandomRingType;
  typedef math_utils::RandomRing<float_v::size() * DigitizationParameters::HitRRSize> PheRandomRingType;

  using ChannelBCDataF = std::array<float, NTimeBinsPerBC>;

 public:
  struct BCCache : public o2::InteractionRecord {
    std::array<ChannelBCDataF, Nchannels> pulse = {};
    std::vector<o2::fdd::MCLabel> labels;

    BCCache();

    void clear()
    {
      for (auto& chan : pulse) {
        chan.fill(0.);
      }
    }

    BCCache& operator=(const o2::InteractionRecord& ir)
    {
      o2::InteractionRecord::operator=(ir);
      return *this;
    }
    void print() const;
  };

  void process(const std::vector<o2::fdd::Hit>& hits,
               std::vector<o2::fdd::Digit>& digitsBC,
               std::vector<o2::fdd::ChannelData>& digitsCh,
               std::vector<o2::fdd::DetTrigInput>& digitsTrig,
               o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>& labels);

  void flush(std::vector<o2::fdd::Digit>& digitsBC,
             std::vector<o2::fdd::ChannelData>& digitsCh,
             std::vector<o2::fdd::DetTrigInput>& digitsTrig,
             o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>& labels);

  void setEventTime(long value) { mEventTime = value; }
  void setEventID(int id) { mEventID = id; }
  void setSrcID(int id) { mSrcID = id; }
  void setInteractionRecord(const o2::InteractionTimeRecord& src) { mIntRecord = src; }

  void setTriggers(o2::fdd::Digit* digit);
  int simulateLightYield(int pmt, int nPhot);
  float simulateTimeCFD(const ChannelBCDataF& pulse);
  float integrateCharge(const ChannelBCDataF& pulse);

  void init();
  void finish();

 private:
  static constexpr int BCCacheMin = -1, BCCacheMax = 10, NBC2Cache = 1 + BCCacheMax - BCCacheMin;

  void createPulse(int nphe, int parID, double timeHit, std::array<o2::InteractionRecord, NBC2Cache> const& cachedIR, int nCachedIR, int channel);

  BCCache& setBCCache(const o2::InteractionRecord& ir);
  BCCache* getBCCache(const o2::InteractionRecord& ir);

  void storeBC(const BCCache& bc,
               std::vector<o2::fdd::Digit>& digitsBC, std::vector<o2::fdd::ChannelData>& digitsCh, std::vector<o2::fdd::DetTrigInput>& digitsTrig,
               o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>& labels);

  long mEventTime;                      // TF (run) timestamp
  o2::InteractionTimeRecord mIntRecord; // Interaction record (orbit, bc) -> InteractionTimeRecord
  int mEventID;                         // ID of the current event
  int mSrcID;                           // signal, background or QED
  std::deque<BCCache> mCache;           // cached BCs data
  o2::fdd::Triggers mTriggers;

  DigitizationParameters parameters;

  ChannelBCDataF mTimeCFD;                                                                  // Time series for CFD measurement
  static constexpr float mBinSize = o2::constants::lhc::LHCBunchSpacingNS / NTimeBinsPerBC; // Time width of the pulse bin - HPTDC resolution
  float mPmtTimeIntegral;

  // Random rings
  HitRandomRingType mRndScintDelay;
  PheRandomRingType mRndGainVar;
  PheRandomRingType mRndSignalShape;

  // 8 tables starting at different sub-bin positions, i.e, [-4:4] / 8 * mBinSize
  // wit each table containg values for start + [-2:2:mBinSize] * DigitizationParameters::mPmtTransitTime
  std::array<std::vector<float>, DigitizationParameters::NResponseTables> mPMResponseTables;

  static Double_t PMResponse(Double_t x);
  static Double_t PMResponse(Double_t* x, Double_t*);
  static Double_t SinglePhESpectrum(Double_t* x, Double_t* par);

  ClassDefNV(Digitizer, 4);
};
} // namespace fdd
} // namespace o2

#endif
