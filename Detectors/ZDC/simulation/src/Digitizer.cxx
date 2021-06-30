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

#include "CommonConstants/LHCConstants.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "Framework/Logger.h"
#include "ZDCSimulation/Digitizer.h"
#include "ZDCSimulation/SimCondition.h"
#include "ZDCSimulation/ZDCSimParam.h"
#include <TRandom.h>

using namespace o2::zdc;

//______________________________________________________________________________
Digitizer::BCCache::BCCache()
{
  memset(&data, 0, NChannels * sizeof(ChannelBCDataF));
}

//______________________________________________________________________________
Digitizer::ModuleConfAux::ModuleConfAux(const Module& md) : id(md.id)
{
  if (md.id < 0 || md.id >= NModules) {
    LOG(FATAL) << "Module id = " << md.id << " not in allowed range [0:" << NModules << ")";
  }
  // construct aux helper from full module description
  for (int ic = Module::MaxChannels; ic--;) {
    if (md.channelID[ic] >= IdDummy) {
      if (md.readChannel[ic]) {
        readChannels |= 0x1 << (NChPerModule * md.id + ic);
      }
      if (md.trigChannel[ic]) {
        trigChannels |= 0x1 << (NChPerModule * md.id + ic);
      }
    }
  }
}

//______________________________________________________________________________
// this will process hits and fill the digit vector with digits which are finalized
void Digitizer::process(const std::vector<o2::zdc::Hit>& hits,
                        std::vector<o2::zdc::BCData>& digitsBC,
                        std::vector<o2::zdc::ChannelData>& digitsCh,
                        o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>& labels)
{
  // loop over all hits and produce digits
  LOG(DEBUG) << "Processing IR = " << mIR << " | NHits = " << hits.size();

  flush(digitsBC, digitsCh, labels); // flush cached signal which cannot be affect by new event

  for (auto& hit : hits) {

    std::array<o2::InteractionRecord, NBC2Cache> cachedIR;
    // hit of given IR can con
    // for each hit find out sector + detector information
    int detID = hit.GetDetectorID();
    int secID = hit.getSector();
    float nPhotons;
    if (detID == ZEM) {
      // ZEM calorimeters have only common PM
      nPhotons = hit.getPMCLightYield();
    } else {
      nPhotons = (secID == Common) ? hit.getPMCLightYield() : hit.getPMQLightYield();
    }
    if (!nPhotons) {
      continue;
    }
    if (nPhotons < 0 || nPhotons > 1e6) {
      int chan = toChannel(detID, secID);
      LOG(ERROR) << "Anomalous number of photons " << nPhotons << " for channel " << chan << '(' << channelName(chan) << ')';
      continue;
    }

    double hTime = hit.GetTime() - getTOFCorrection(detID); // account for TOF to detector
    hTime += mIR.getTimeNS();
    //
    o2::InteractionRecord irHit(hTime); // BC in which the hit appears (might be different from interaction BC for slow particles)

    // nominal time of the BC to which the hit will be attributed
    double bcTime = o2::InteractionRecord::bc2ns(irHit.bc, irHit.orbit);

    int nCachedIR = 0;
    for (int i = BCCacheMin; i < BCCacheMax + 1; i++) {
      double tNS = hTime + o2::constants::lhc::LHCBunchSpacingNS * i;
      cachedIR[nCachedIR].setFromNS(tNS);
      if (tNS < 0 && cachedIR[nCachedIR] > irHit) {
        continue; // don't go to negative BC/orbit (it will wrap)
      }
      getCreateBCCache(cachedIR[nCachedIR++]); // ensure existence of cached container
    }
    auto channel = toChannel(detID, secID);
    phe2Sample(nPhotons, hit.getParentID(), hTime, cachedIR, nCachedIR, channel);
    // if digit for this sector does not exist, create one otherwise add to it
  }
}

//______________________________________________________________________________
void Digitizer::flush(std::vector<o2::zdc::BCData>& digitsBC,
                      std::vector<o2::zdc::ChannelData>& digitsCh,
                      o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>& labels)
{
  // do we have something to flush? We can do this only for cached BC data which is distanced from currently processed BC by NBCReadOut
  int lastDoneBCid = -1, diff2Last = 0;
  int nCached = mCache.size();
  if (nCached < 1) {
    return;
  }
  if (mIR.differenceInBC(mCache.back()) > -BCCacheMin) {
    LOG(DEBUG) << "Generating new pedestal BL fluct. for BC range " << mCache.front() << " : " << mCache.back();
    generatePedestal();
  } else {
    return;
  }
  o2::InteractionRecord ir0(mCache.front());
  int cacheSpan = 1 + mCache.back().differenceInBC(ir0);
  LOG(DEBUG) << "Cache spans " << cacheSpan << " with " << nCached << " BCs cached";

  mFastCache.clear();
  mFastCache.resize(cacheSpan, nullptr);
  mStoreChanMask.clear();
  mStoreChanMask.resize(cacheSpan + mNBCAHead, 0);

  for (int ibc = nCached; ibc--;) { // digitize BCs which might not be affected by future events
    auto& bc = mCache[ibc];
    lastDoneBCid = ibc;
    if (!bc.digitized) {
      digitizeBC(bc);
    }
    int bcSlot = mCache[ibc].differenceInBC(ir0);
    mFastCache[bcSlot] = &mCache[ibc]; // add to fast access cache
  }
  mDummyBC.clear();
  digitizeBC(mDummyBC);

  // check trigger condition for digitized BCs
  for (int ibc = 0; ibc < cacheSpan; ibc++) {
    if (mFastCache[ibc] && !mFastCache[ibc]->triggerChecked) {
      triggerBC(ibc);
    }
  } // all allowed BCs are checked for trigger

  // store triggered BC with requested few BCs ahead
  for (int ibc = -mNBCAHead; ibc < cacheSpan; ibc++) {
    auto bcr = mStoreChanMask[ibc + mNBCAHead];
    if (!bcr) {
      continue; // nothing to store for this BC
    }
    BCCache* bcPtr = nullptr;
    if (ibc < 0 || !mFastCache[ibc]) {
      mDummyBC = ir0 + ibc; // fix the IR of the dummy BC
      bcPtr = &mDummyBC;
    } else {
      bcPtr = mFastCache[ibc];
    }
    storeBC(*bcPtr, bcr, digitsBC, digitsCh, labels);
  } // all allowed BCs are checked for trigger

  // clean cache for BCs which are not needed anymore
  LOG(DEBUG) << "Cleaning cache";
  mCache.erase(mCache.begin(), mCache.end());
}

//______________________________________________________________________________
void Digitizer::generatePedestal()
{
  for (int idet : {ZNA, ZPA, ZNC, ZPC}) {
    int chanSum = toChannel(idet, Sum);
    mPedestalBLFluct[chanSum] = 0.;
    int comm = toChannel(idet, Common);
    mPedestalBLFluct[comm] = gRandom->Gaus(0, mSimCondition->channels[comm].pedestalFluct);
    for (int ic : {Ch1, Ch2, Ch3, Ch4}) {
      int chan = toChannel(idet, ic);
      mPedestalBLFluct[chanSum] += mPedestalBLFluct[chan] = gRandom->Gaus(0, mSimCondition->channels[chan].pedestalFluct);
    }
  }
  mPedestalBLFluct[IdZEM1] = gRandom->Gaus(0, mSimCondition->channels[IdZEM1].pedestalFluct);
  mPedestalBLFluct[IdZEM2] = gRandom->Gaus(0, mSimCondition->channels[IdZEM2].pedestalFluct);
}

//______________________________________________________________________________
void Digitizer::digitizeBC(BCCache& bc)
{
  auto& bcdata = bc.data;
  auto& bcdigi = bc.digi;
  // apply gain
  for (int idet : {ZNA, ZPA, ZNC, ZPC}) {
    for (int ic : {Common, Ch1, Ch2, Ch3, Ch4}) {
      int chan = toChannel(idet, ic);
      auto gain = mSimCondition->channels[chan].gain;
      for (int ib = NTimeBinsPerBC; ib--;) {
        bcdata[chan][ib] *= gain;
      }
    }
  }
  for (int ib = NTimeBinsPerBC; ib--;) {
    bcdata[IdZEM1][ib] *= mSimCondition->channels[IdZEM1].gain;
    bcdata[IdZEM2][ib] *= mSimCondition->channels[IdZEM2].gain;
  }
  // Prepare sum of towers before adding noise
  for (int ib = NTimeBinsPerBC; ib--;) {
    bcdata[IdZNASum][ib] = mSimCondition->channels[IdZNASum].gain *
                           (bcdata[IdZNA1][ib] + bcdata[IdZNA2][ib] + bcdata[IdZNA3][ib] + bcdata[IdZNA4][ib]);
    bcdata[IdZPASum][ib] = mSimCondition->channels[IdZPASum].gain *
                           (bcdata[IdZPA1][ib] + bcdata[IdZPA2][ib] + bcdata[IdZPA3][ib] + bcdata[IdZPA4][ib]);
    bcdata[IdZNCSum][ib] = mSimCondition->channels[IdZNCSum].gain *
                           (bcdata[IdZNC1][ib] + bcdata[IdZNC2][ib] + bcdata[IdZNC3][ib] + bcdata[IdZNC4][ib]);
    bcdata[IdZPCSum][ib] = mSimCondition->channels[IdZPCSum].gain *
                           (bcdata[IdZPC1][ib] + bcdata[IdZPC2][ib] + bcdata[IdZPC3][ib] + bcdata[IdZPC4][ib]);
  }
  // Digitize the signals connected to each channel of the different modules
  for (const auto& md : mModuleConfig->modules) {
    if (md.id >= 0 && md.id < NModules) {
      for (int ic = Module::MaxChannels; ic--;) {
        int id = md.channelID[ic];
        if (id >= 0 && id < NChannels) {
          const auto& chanConf = mSimCondition->channels[id];
          auto pedBaseLine = mSimCondition->channels[id].pedestal;
          // Geographical position of signal in the setup
          auto ipos = NChPerModule * md.id + ic;
          for (int ib = NTimeBinsPerBC; ib--;) {
            bcdigi[ipos][ib] = bcdata[id][ib] + gRandom->Gaus(pedBaseLine + mPedestalBLFluct[id], chanConf.pedestalNoise);
            int adc = std::nearbyint(bcdigi[ipos][ib]);
            bcdigi[ipos][ib] = adc < ADCMax ? (adc > ADCMin ? adc : ADCMin) : ADCMax;
          }
          LOG(DEBUG) << "md " << md.id << " ch " << ic << " sig " << id << " " << ChannelNames[id]
                     << bcdigi[ipos][0] << " " << bcdigi[ipos][1] << " " << bcdigi[ipos][2] << " " << bcdigi[ipos][3] << " " << bcdigi[ipos][4] << " " << bcdigi[ipos][5] << " "
                     << bcdigi[ipos][6] << " " << bcdigi[ipos][7] << " " << bcdigi[ipos][8] << " " << bcdigi[ipos][9] << " " << bcdigi[ipos][10] << " " << bcdigi[ipos][11];
        }
      }
    }
  }
  bc.digitized = true;
}

//______________________________________________________________________________
bool Digitizer::triggerBC(int ibc)
{
  // check trigger for the cached BC in the position ibc
  auto& bcCached = *mFastCache[ibc];

  LOG(DEBUG) << "CHECK TRIGGER " << ibc << " IR=" << bcCached;

  // Check trigger condition regardless of run type, will apply later the trigger mask
  for (const auto& md : mModuleConfig->modules) {
    if (md.id >= 0 && md.id < NModules) {
      for (int ic = Module::MaxChannels; ic--;) {
        //int id=md.channelID[ic];
        auto trigCh = md.trigChannelConf[ic];
        int id = trigCh.id;
        if (id >= 0 && id < NChannels) {
          auto ipos = NChPerModule * md.id + ic;
          int last1 = trigCh.last + 2;
          bool okPrev = false;
#ifdef ZDC_DOUBLE_TRIGGER_CONDITION
          // look for 2 consecutive bins (the 1st one spanning trigCh.first : trigCh.last range) so that
          // signal[bin]-signal[bin+trigCh.shift] > trigCh.threshold
          for (int ib = trigCh.first; ib < last1; ib++) { // ib may be negative, so we shift by offs and look in the ADC cache
            int binF, bcFidx = ibc + binHelper(ib, binF);
            int binL, bcLidx = ibc + binHelper(ib + trigCh.shift, binL);
            const auto& bcF = (bcFidx < 0 || !mFastCache[bcFidx]) ? mDummyBC : *mFastCache[bcFidx];
            const auto& bcL = (bcLidx < 0 || !mFastCache[bcLidx]) ? mDummyBC : *mFastCache[bcLidx];
            bool ok = bcF.digi[ipos][binF] - bcL.digi[ipos][binL] > trigCh.threshold;
            if (ok && okPrev) {                                            // trigger ok!
              bcCached.trigChanMask |= 0x1 << (NChPerModule * md.id + ic); // register trigger mask
              LOG(DEBUG) << bcF.digi[ipos][binF] << " - " << bcL.digi[ipos][binL] << " = " << bcF.digi[ipos][binF] - bcL.digi[ipos][binL] << " > " << trigCh.threshold;
              LOG(DEBUG) << " hit [" << md.id << "," << ic << "] " << int(id) << "(" << ChannelNames[id] << ") => " << bcCached.trigChanMask;
              break;
            }
            okPrev = ok;
          }
#else
          bool okPPrev = false;
          // look for 3 consecutive bins (the 1st one spanning trigCh.first : trigCh.last range) so that
          // signal[bin]-signal[bin+trigCh.shift] > trigCh.threshold
          for (int ib = trigCh.first - 1; ib < last1; ib++) { // ib may be negative, so we shift by offs and look in the ADC cache
            int binF, bcFidx = ibc + binHelper(ib, binF);
            int binL, bcLidx = ibc + binHelper(ib + trigCh.shift, binL);
            const auto& bcF = (bcFidx < 0 || !mFastCache[bcFidx]) ? mDummyBC : *mFastCache[bcFidx];
            const auto& bcL = (bcLidx < 0 || !mFastCache[bcLidx]) ? mDummyBC : *mFastCache[bcLidx];
            bool ok = bcF.digi[ipos][binF] - bcL.digi[ipos][binL] > trigCh.threshold;
            if (ok && okPrev && okPPrev) {                                 // trigger ok!
              bcCached.trigChanMask |= 0x1 << (NChPerModule * md.id + ic); // register trigger mask
              LOG(DEBUG) << bcF.digi[ipos][binF] << " - " << bcL.digi[ipos][binL] << " = " << bcF.digi[ipos][binF] - bcL.digi[ipos][binL] << " > " << trigCh.threshold;
              LOG(DEBUG) << " hit [" << md.id << "," << ic << "] " << int(id) << "(" << ChannelNames[id] << ") => " << bcCached.trigChanMask;
              break;
            }
            okPPrev = okPrev;
            okPrev = ok;
          }
#endif
        }
      }
    }
  }

  // just check if this BC IR corresponds to external trigger
  if (!mIRExternalTrigger.empty() && mIRExternalTrigger.front() == bcCached) {
    bcCached.extTrig = ALICETriggerMask;
    mIRExternalTrigger.pop_front(); // suppress accounted external trigger
  }
  // This works in autotrigger mode, partially for triggered mode
  if (bcCached.trigChanMask & mTriggerableChanMask) { // there are triggered channels, flag modules/channels to read
    for (int ibcr = ibc - mNBCAHead; ibcr <= ibc; ibcr++) {
      auto& bcr = mStoreChanMask[ibcr + mNBCAHead];
      for (const auto& mdh : mModConfAux) {
        if (bcCached.trigChanMask & mdh.trigChannels) { // are there triggered channels in this module?
          bcr |= mdh.readChannels;                      // flag channels to store
        }
      }
    }
  }
  bcCached.triggerChecked = true;
  return bcCached.trigChanMask;
}

//______________________________________________________________________________
void Digitizer::storeBC(const BCCache& bc, uint32_t chan2Store,
                        std::vector<o2::zdc::BCData>& digitsBC, std::vector<o2::zdc::ChannelData>& digitsCh,
                        o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>& labels)
{
  // store selected data of selected BC
  if (!chan2Store) {
    return;
  }
  LOG(DEBUG) << "Storing ch: " << chanPattern(chan2Store) << " trigger: " << chanPattern(bc.trigChanMask) << " for BC " << bc;

  int first = digitsCh.size(), nSto = 0;
  for (const auto& md : mModuleConfig->modules) {
    if (md.id >= 0 && md.id < NModules) {
      for (int ic = 0; ic < Module::MaxChannels; ic++) {
        auto ipos = NChPerModule * md.id + ic;
        if (chan2Store & (0x1 << ipos)) {
          digitsCh.emplace_back(md.channelID[ic], bc.digi[ipos]);
          nSto++;
        }
      }
    }
  }

  int nBC = digitsBC.size();
  digitsBC.emplace_back(first, nSto, bc, chan2Store, bc.trigChanMask, bc.extTrig);
  // TODO clarify if we want to store MC labels for all channels or only for stored ones
  if (!mSkipMCLabels) {
    for (const auto& lbl : bc.labels) {
      if (chan2Store & (0x1 << lbl.getChannel())) {
        labels.addElement(nBC, lbl);
      }
    }
  }
}

//______________________________________________________________________________
void Digitizer::phe2Sample(int nphe, int parID, double timeHit, std::array<o2::InteractionRecord, NBC2Cache> const& cachedIR, int nCachedIR, int channel)
{
  //function to simulate the waveform from no. of photoelectrons seen in a given sample
  // for electrons at timeInSample wrt beginning of the sample

  double time0 = cachedIR[0].bc2ns(); // start time of the 1st cashed BC
  const auto& chanConfig = mSimCondition->channels[channel];

  float timeDiff = time0 - timeHit;
  int sample = (timeDiff - gRandom->Gaus(chanConfig.timePosition, chanConfig.timeJitter)) * ChannelSimCondition::ShapeBinWidthInv + chanConfig.ampMinID;
  int ir = 0;
  bool stop = false;

  do {
    auto bcCache = getBCCache(cachedIR[ir]);
    bool added = false;
    for (int ib = 0; ib < NTimeBinsPerBC; ib++) {
      if (sample >= chanConfig.shape.size()) {
        stop = true;
        break;
      }
      if (sample >= 0) {
        auto signal = chanConfig.shape[sample] * nphe; // signal not accounting for the gain
        (*bcCache).data[channel][ib] += signal;
        added = true;
      }
      sample += o2::constants::lhc::LHCBunchSpacingNS / NTimeBinsPerBC * ChannelSimCondition::ShapeBinWidthInv; // sample increment for next bin
    }
    if (added) {
      (*bcCache).labels.emplace_back(parID, mEventID, mSrcID, channel);
    }
  } while (++ir < nCachedIR && !stop);
}

//______________________________________________________________________________
o2::zdc::Digitizer::BCCache& Digitizer::getCreateBCCache(const o2::InteractionRecord& ir)
{
  if (mCache.empty() || mCache.back() < ir) {
    mCache.emplace_back();
    auto& cb = mCache.back();
    cb = ir;
    return cb;
  }
  if (mCache.front() > ir) {
    mCache.emplace_front();
    auto& cb = mCache.front();
    cb = ir;
    return cb;
  }

  for (auto cb = mCache.begin(); cb != mCache.end(); cb++) {
    if ((*cb) == ir) {
      return *cb;
    }
    if (ir < (*cb)) {
      auto cbnew = mCache.emplace(cb); // insert new element before cb
      (*cbnew) = ir;
      return (*cbnew);
    }
  }
  return mCache.front();
}

//______________________________________________________________________________
o2::zdc::Digitizer::BCCache* Digitizer::getBCCache(const o2::InteractionRecord& ir)
{
  // get pointer on existing cache
  for (auto cb = mCache.begin(); cb != mCache.end(); cb++) {
    if ((*cb) == ir) {
      return &(*cb);
    }
  }
  return nullptr;
}

//______________________________________________________________________________
void Digitizer::init()
{
  if (mCCDBServer.empty()) {
    LOG(FATAL) << "ZDC digitizer: CCDB server is not set";
  }
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(mCCDBServer);

  auto& sopt = ZDCSimParam::Instance();
  mIsContinuous = sopt.continuous;
  mNBCAHead = mIsContinuous ? sopt.nBCAheadCont : sopt.nBCAheadTrig;
  LOG(INFO) << "Initialized in " << (mIsContinuous ? "Cont" : "Trig") << " mode, " << mNBCAHead
            << " BCs will be stored ahead of Trigger";
  LOG(INFO) << "Trigger bit masking is " << (mMaskTriggerBits ? "ON (default)" : "OFF (debugging)");
  LOG(INFO) << "MC Labels are " << (mSkipMCLabels ? "SKIPPED" : "SAVED (default)");
}

//______________________________________________________________________________
void Digitizer::refreshCCDB()
{
  // fetch ccdb objects. TODO: decide if this stays here or goes to the Spec
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  if (mTimeStamp == mgr.getTimestamp()) {
    return;
  }
  mgr.setTimestamp(mTimeStamp);

  if (!mModuleConfig) { // load this only once
    mModuleConfig = mgr.get<ModuleConfig>(CCDBPathConfigModule);
    LOG(INFO) << "Loaded module configuration for timestamp " << mTimeStamp;
    // fetch trigger info
    mTriggerConfig.clear();
    mModConfAux.clear();
    for (const auto& md : mModuleConfig->modules) {
      if (md.id >= 0 && md.id < NModules) {
        mModConfAux.emplace_back(md);
        for (int ic = Module::MaxChannels; ic--;) {
          // We consider all channels that can produce a hit
          if (md.trigChannel[ic] || (md.trigChannelConf[ic].shift > 0 && md.trigChannelConf[ic].threshold > 0)) {
            const auto& trgChanConf = md.trigChannelConf[ic];
            if (trgChanConf.last + trgChanConf.shift + 1 >= NTimeBinsPerBC) {
              LOG(ERROR) << "Wrong trigger settings";
            }
            mTriggerConfig.emplace_back(trgChanConf);
            // We insert in the trigger mask only the channels that are actually triggering
            // Trigger mask is geographical, bit position is relative to the module and channel
            // where signal is connected
            if (md.trigChannel[ic]) {
              LOG(INFO) << "Adding channel [" << md.id << "," << ic << "] " << int(trgChanConf.id) << '(' << channelName(trgChanConf.id) << ") as triggering one";
              // TODO insert check if bit is already used. Should never happen
              mTriggerableChanMask |= 0x1 << (NChPerModule * md.id + ic);
            } else {
              LOG(INFO) << "Adding channel [" << md.id << "," << ic << "] " << int(trgChanConf.id) << '(' << channelName(trgChanConf.id) << ") as discriminator";
            }
            if (trgChanConf.first < mTrigBinMin) {
              mTrigBinMin = trgChanConf.first;
            }
            if (trgChanConf.last + trgChanConf.shift > mTrigBinMax) {
              mTrigBinMax = trgChanConf.last + trgChanConf.shift;
            }
          }
          if (md.feeID[ic] < 0 || md.feeID[ic] >= NLinks) {
            LOG(FATAL) << "FEEID " << md.feeID[ic] << " not in allowed range [0:" << NLinks << ")";
          }
        }
      } else {
        LOG(FATAL) << "Module id: " << md.id << " is out of range";
      }
    }
    mModuleConfig->print();
  }

  if (!mSimCondition) { // load this only once
    mSimCondition = mgr.get<SimCondition>(CCDBPathConfigSim);
    LOG(INFO) << "Loaded simulation configuration for timestamp " << mTimeStamp;
    mSimCondition->print();
  }

  setTriggerMask();
  setReadoutMask();
}

//______________________________________________________________________________
void Digitizer::BCCache::print() const
{
  std::bitset<NChannels> tmsk(trigChanMask);
  printf("Cached Orbit:%5d/BC:%4d | digitized:%d  triggerChecked:%d (trig.: %s)\n",
         orbit, bc, digitized, triggerChecked, tmsk.to_string().c_str());
  for (int ic = 0; ic < NChannels; ic++) {
    printf("Ch[%d](%s) | ", ic, channelName(ic));
    for (int ib = 0; ib < NTimeBinsPerBC; ib++) {
      printf("%+8.1f ", data[ic][ib]);
    }
    printf("\n");
  }
}

//______________________________________________________________________________
void Digitizer::setTriggerMask()
{
  mTriggerMask = 0;
  std::string printTriggerMask{};

  for (int im = 0; im < NModules; im++) {
    if (im > 0) {
      printTriggerMask += " ";
    }
    printTriggerMask += std::to_string(im);
    printTriggerMask += "[";
    for (int ic = 0; ic < NChPerModule; ic++) {
      if (mModuleConfig->modules[im].trigChannel[ic]) {
        uint32_t tmask = 0x1 << (im * NChPerModule + ic);
        mTriggerMask = mTriggerMask | tmask;
        printTriggerMask += "T";
      } else {
        printTriggerMask += " ";
      }
    }
    printTriggerMask += "]";
    uint32_t mytmask = mTriggerMask >> (im * NChPerModule);
    LOGF(INFO, "Trigger mask for module %d 0123 %c%c%c%c\n", im,
         mytmask & 0x1 ? 'T' : 'N', mytmask & 0x2 ? 'T' : 'N', mytmask & 0x4 ? 'T' : 'N', mytmask & 0x8 ? 'T' : 'N');
  }
  LOGF(INFO, "TriggerMask=0x%08x %s\n", mTriggerMask, printTriggerMask.c_str());
}

//______________________________________________________________________________
void Digitizer::setReadoutMask()
{
  mReadoutMask = 0;
  std::string printReadoutMask{};

  for (int im = 0; im < NModules; im++) {
    if (im > 0) {
      printReadoutMask += " ";
    }
    printReadoutMask += std::to_string(im);
    printReadoutMask += "[";
    for (int ic = 0; ic < NChPerModule; ic++) {
      if (mModuleConfig->modules[im].readChannel[ic]) {
        uint32_t rmask = 0x1 << (im * NChPerModule + ic);
        mReadoutMask = mReadoutMask | rmask;
        printReadoutMask += "R";
      } else {
        printReadoutMask += " ";
      }
    }
    printReadoutMask += "]";
    uint32_t myrmask = mReadoutMask >> (im * NChPerModule);
    LOGF(INFO, "Readout mask for module %d 0123 %c%c%c%c\n", im,
         myrmask & 0x1 ? 'R' : 'N', myrmask & 0x2 ? 'R' : 'N', myrmask & 0x4 ? 'R' : 'N', myrmask & 0x8 ? 'R' : 'N');
  }
  LOGF(INFO, "ReadoutMask=0x%08x %s\n", mReadoutMask, printReadoutMask.c_str());
}

//______________________________________________________________________________
void Digitizer::assignTriggerBits(uint32_t ibc, std::vector<BCData>& bcData)
{
  // Triggers refer to the HW trigger conditions (32 possible channels)

  uint32_t nbcTot = bcData.size();
  auto& currBC = bcData[ibc];

  for (int is = -1; is < 4; is++) {
    int ibc_peek = ibc + is;
    if (ibc_peek < 0) {
      continue;
    }
    if (ibc_peek >= nbcTot) {
      break;
    }
    const auto& otherBC = bcData[ibc_peek];
    auto diffBC = otherBC.ir.differenceInBC(currBC.ir);
    if (diffBC < -1) {
      continue;
    }
    if (diffBC > 3) {
      break;
    }
    if (otherBC.ext_triggers && diffBC >= 0) {
      for (int im = 0; im < NModules; im++) {
        currBC.moduleTriggers[im] |= 0x1 << diffBC;
      }
    }
    if (otherBC.triggers) {
      // Assign trigger bits in payload
      for (int im = 0; im < NModules; im++) {
        uint32_t tmask = (0xf << (im * NChPerModule)) & mTriggerMask;
        if (otherBC.triggers & tmask) {
          currBC.moduleTriggers[im] |= 0x1 << (5 + diffBC);
        }
      }
    }
  }
  // Printout before cleanup
  //currBC.print(mTriggerMask);
}

void Digitizer::Finalize(std::vector<BCData>& bcData, std::vector<o2::zdc::OrbitData>& pData)
{
  // Compute scalers for digitized data (works only for triggering channels)
  for (Int_t ipd = 0; ipd < pData.size(); ipd++) {
    for (int id = 0; id < NChannels; id++) {
      pData[ipd].scaler[id] = 0;
    }
  }
  for (int im = 0; im < NModules; im++) {
    for (int ic = 0; ic < NChPerModule; ic++) {
      uint32_t mask = 0x1 << (im * NChPerModule + ic);
      auto id = mModuleConfig->modules[im].channelID[ic];
      for (uint32_t ibc = 0; ibc < bcData.size(); ibc++) {
        auto& currBC = bcData[ibc];
        if ((currBC.triggers & mask) && (mReadoutMask & mask)) {
          for (Int_t ipd = 0; ipd < pData.size(); ipd++) {
            if (pData[ipd].ir.orbit == currBC.ir.orbit) {
              pData[ipd].scaler[id]++;
            }
          }
        }
      }
    }
  }
  // Default is to mask trigger bits, we can leave them for debugging
  if (mMaskTriggerBits) {
    for (uint32_t ibc = 0; ibc < bcData.size(); ibc++) {
      auto& currBC = bcData[ibc];
      for (int im = 0; im < NModules; im++) {
        // Cleanup hits if module has no trigger
        uint32_t mmask = 0xf << im;
        if ((currBC.triggers & mTriggerMask & mmask) == 0) {
          currBC.triggers = currBC.triggers & (~mmask);
        }
      }
      // Cleanup trigger bits for channels that are not readout
      currBC.triggers &= mReadoutMask;
      // Printout after cleanup
      //currBC.print(mTriggerMask);
    }
  }
}

//______________________________________________________________________________
void Digitizer::findEmptyBunches(const std::bitset<o2::constants::lhc::LHCMaxBunches>& bunchPattern)
{
  mNEmptyBCs = 0;
  for (int ib = 0; ib < o2::constants::lhc::LHCMaxBunches; ib++) {
    int mb = (ib + 31) % o2::constants::lhc::LHCMaxBunches;                                                 // beam gas from back of calorimeter
    int m1 = ib ? ((ib - 1) % o2::constants::lhc::LHCMaxBunches) : (o2::constants::lhc::LHCMaxBunches - 1); // previous bunch
    int cb = ib;                                                                                            // current bunch crossing
    int p1 = (ib + 1) % o2::constants::lhc::LHCMaxBunches;                                                  // colliding + 1
    int p2 = (ib + 2) % o2::constants::lhc::LHCMaxBunches;                                                  // colliding + 2
    int p3 = (ib + 3) % o2::constants::lhc::LHCMaxBunches;                                                  // colliding + 3
    if (!(bunchPattern[mb] || bunchPattern[m1] || bunchPattern[cb] || bunchPattern[p1] || bunchPattern[p2] || bunchPattern[p3])) {
      mNEmptyBCs++;
    }
  }
  LOG(INFO) << "There are " << mNEmptyBCs << " clean empty bunches";
}

//______________________________________________________________________________
void Digitizer::updatePedestalReference(OrbitData& pdata)
{
  // Compute or update baseline reference
  for (uint32_t id = 0; id < NChannels; id++) {
    auto base_m = mSimCondition->channels[id].pedestal;                                                   // Average pedestal
    auto base_s = mSimCondition->channels[id].pedestalFluct;                                              // Baseline oscillations
    auto base_n = mSimCondition->channels[id].pedestalNoise;                                              // Electronic noise
    float ped = gRandom->Gaus(12. * mNEmptyBCs * base_m, 12. * 2. * base_s * std::sqrt(0.5 * mNEmptyBCs)) // 2 for fluctuation every 2 BCs
                + gRandom->Gaus(0, base_n * std::sqrt(12. * mNEmptyBCs));
    int16_t peds = std::round(8. * ped / mNEmptyBCs / 12.);
    if (peds < SHRT_MIN) {
      peds = SHRT_MIN;
    } else if (peds > SHRT_MAX) {
      peds = SHRT_MAX;
    }
    pdata.data[id] = peds;
  }
}
