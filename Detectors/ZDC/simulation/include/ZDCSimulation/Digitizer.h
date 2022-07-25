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

#ifndef DETECTORS_ZDC_DIGITIZER_H_
#define DETECTORS_ZDC_DIGITIZER_H_

#include "DataFormatsZDC/Hit.h" // for the hit
#include "DataFormatsZDC/MCLabel.h"
#include "ZDCBase/ModuleConfig.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <vector>
#include <array>
#include <deque>
#include <bitset>

namespace o2
{
namespace zdc
{

class SimCondition;

class Digitizer
{
  using ChannelBCDataF = std::array<float, NTimeBinsPerBC>;

 public:
  struct BCCache : public o2::InteractionRecord {
    std::array<ChannelBCDataF, NChannels> data = {};
    std::array<ChannelBCDataF, NDigiChannels> digi = {};
    std::vector<o2::zdc::MCLabel> labels;
    bool digitized = false;
    bool triggerChecked = false;
    uint32_t trigChanMask = 0; // mask of triggered channels
    uint8_t extTrig = 0;       // external trigger
    static constexpr uint32_t AllChannelsMask = 0x80000000;

    BCCache();

    void clear()
    {
      digitized = false;
      triggerChecked = false;
      trigChanMask = 0;
      for (auto& chan : data) {
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

  struct ModuleConfAux {
    int id = 0;
    uint32_t readChannels = 0; // channels to read
    uint32_t trigChannels = 0; // trigger channels
    ModuleConfAux() = default;
    ModuleConfAux(const Module& md);
  };

  void init();

  // set event time
  void setEventID(int eventID) { mEventID = eventID; }
  void setSrcID(int sID) { mSrcID = sID; }
  void setInteractionRecord(const o2::InteractionTimeRecord& ir) { mIR = ir; }
  void setTimeStamp(long t) { mTimeStamp = t; }

  void addTriggeredBC(const o2::InteractionRecord& ir)
  {
    // RS TODO: this method should be called e.g. from the steering DPL device in the triggered mode
    if (mIRExternalTrigger.empty() || mIRExternalTrigger.back() != ir) {
      mIRExternalTrigger.push_back(ir); // add new trigger if there was no already trigger in the same BC
    }
  }

  void process(const std::vector<o2::zdc::Hit>& hits,
               std::vector<o2::zdc::BCData>& digitsBC,
               std::vector<o2::zdc::ChannelData>& digitsCh,
               o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>& labels);

  void flush(std::vector<o2::zdc::BCData>& digitsBC,
             std::vector<o2::zdc::ChannelData>& digitsCh,
             o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>& labels);

  // no setters: initialization done via refreshCCDB
  const SimCondition* getSimCondition() const { return mSimCondition; }
  const ModuleConfig* getModuleConfig() const { return mModuleConfig; }

  void setContinuous(bool v = true) { mIsContinuous = v; }
  bool isContinuous() const { return mIsContinuous; }
  void updatePedestalReference(OrbitData& pdata);
  void refreshCCDB();
  void setCCDBServer(const std::string& s) { mCCDBServer = s; }
  void findEmptyBunches(const std::bitset<o2::constants::lhc::LHCMaxBunches>& bunchPattern);
  int getNEmptyBunches() const { return mNEmptyBCs; }
  void assignTriggerBits(uint32_t ibc, std::vector<BCData>& bcData);                  // Assign trigger bits for nearby bunch crossings
  void Finalize(std::vector<BCData>& bcData, std::vector<o2::zdc::OrbitData>& pData); // Mask trigger bits for current bunch crossing
  void setMaskTriggerBits(bool v = true) { mMaskTriggerBits = v; }
  bool getMaskTriggerBits() { return mMaskTriggerBits; }
  void setSkipMCLabels(bool v = true) { mSkipMCLabels = v; }
  bool getSkipMCLabels() { return mSkipMCLabels; }

 private:
  static constexpr int BCCacheMin = -1, BCCacheMax = 5, NBC2Cache = 1 + BCCacheMax - BCCacheMin;

  std::bitset<NChannels> chanPattern(uint32_t v) const
  {
    return std::bitset<NChannels>(v);
  }
  void phe2Sample(int nphe, int parID, double timeHit, std::array<o2::InteractionRecord, NBC2Cache> const& cachedIR, int nCachedIR, int channel);
  BCCache& getCreateBCCache(const o2::InteractionRecord& ir);
  BCCache* getBCCache(const o2::InteractionRecord& ir);
  void setTriggerMask();
  void setReadoutMask();
  void generatePedestal();
  void digitizeBC(BCCache& bc);
  bool triggerBC(int ibc);
  void storeBC(const BCCache& bc, uint32_t chan2Store,
               std::vector<o2::zdc::BCData>& digitsBC, std::vector<o2::zdc::ChannelData>& digitsCh,
               o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>& labels);

  bool mIsContinuous = true; // continuous (self-triggered) or externally-triggered readout
  int mEventID = 0;
  int mSrcID = 0;
  long mTimeStamp = 0; // TF (run) timestamp
  o2::InteractionTimeRecord mIR;
  std::deque<o2::InteractionRecord> mIRExternalTrigger; // IRs of externally provided triggered (at the moment MC sampled interactions)

  std::deque<BCCache> mCache;                                    // cached BCs data
  std::array<std::vector<int16_t>, NChannels> mTrigChannelsData; // buffer for fast access to triggered channels data
  int mTrigBinMin = 0xffff;                                      // prefetched min and max
  int mTrigBinMax = -0xffff;                                     // bins to be checked for trigger
  int mNBCAHead = 0;                                             // when storing triggered BC, store also mNBCAHead BCs
  uint32_t mTriggerMask = 0;                                     // Trigger mask from ModuleConfig
  uint32_t mReadoutMask = 0;                                     // Readout mask from ModuleConfig
  int32_t mNEmptyBCs = -1;                                       // Number of clean empty bunches for pedestal evaluation
  float mPedFactor = 1;                                          // Pedestal scaling factor in digitization
  bool mMaskTriggerBits = true;                                  // Mask trigger bits with readout mask
  bool mSkipMCLabels = false;                                    // Skip MC labels in output

  std::string mCCDBServer = "";
  const SimCondition* mSimCondition = nullptr;      ///< externally set SimCondition
  const ModuleConfig* mModuleConfig = nullptr;      ///< externally set ModuleConfig
  std::vector<TriggerChannelConfig> mTriggerConfig; ///< triggering channels
  uint32_t mTriggerableChanMask = 0;                ///< mask of digital discriminators that can actually trigger a module
  std::vector<ModuleConfAux> mModConfAux;           ///< module check helper
  std::vector<BCCache*> mFastCache;                 ///< for the fast iteration over cached BCs + dummy
  std::vector<uint32_t> mStoreChanMask;             ///< pattern of channels to store
  BCCache mDummyBC;                                 ///< dummy BC
  std::array<float, NChannels> mPedestalBLFluct;    ///< pedestal baseline fluctuations per channel for currently digitized cache

  // helper for inspection of bins in preceding or following BC: if we are at the cached BC = I and we need to
  // inspect the bin = ib, we have to look in the binInSiftedBC of BC I-binHelper(ib,binInSiftedBC);
  static constexpr int binHelper(int ib, int& binInSiftedBC)
  {
    if (ib < 0) {
      binInSiftedBC = NTimeBinsPerBC - 1 + (1 + ib) % NTimeBinsPerBC;
      return (ib - (NTimeBinsPerBC - 1)) / NTimeBinsPerBC;
    } else {
      binInSiftedBC = ib % NTimeBinsPerBC;
      return ib / NTimeBinsPerBC;
    }
  }
  ClassDefNV(Digitizer, 1);
};
} // namespace zdc
} // namespace o2

#endif /* DETECTORS_ZDC_DIGITIZER_H_ */
