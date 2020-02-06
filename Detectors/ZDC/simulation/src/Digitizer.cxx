// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCSimulation/Digitizer.h"
#include "ZDCSimulation/Hit.h"
#include "ZDCSimulation/SimCondition.h"
#include "ZDCSimulation/ZDCSimParam.h"
#include <TRandom.h>
#include <FairLogger.h>

using namespace o2::zdc;

Digitizer::BCCache::BCCache()
{
  memset(&data, 0, NChannels * sizeof(ChannelBCDataF));
}

Digitizer::ModuleConfAux::ModuleConfAux(const Module& md) : id(md.id)
{
  // construct aux helper from full module description
  for (int ic = Module::MaxChannels; ic--;) {
    if (md.channelID[ic] >= IdDummy) {
      if (md.readChannel[ic]) {
        readChannels |= 0x1 << md.channelID[ic];
      }
      if (md.trigChannel[ic]) {
        trigChannels |= 0x1 << md.channelID[ic];
      }
    }
  }
}

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
    if (detID == ZEM) { // TODO: ZEMCh1 and Common are both 0, could skip the check for detID
      nPhotons = (secID == ZEMCh1) ? hit.getPMCLightYield() : hit.getPMQLightYield();
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
    hTime += mIR.timeNS;
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

void Digitizer::digitizeBC(BCCache& bc)
{
  auto& bcdata = bc.data;
  // apply gain
  for (int idet : {ZNA, ZPA, ZNC, ZPC}) {
    for (int ic : {Ch1, Ch2, Ch3, Ch4}) {
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
  //
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
  for (int chan = NChannels; chan--;) {
    const auto& chanConf = mSimCondition->channels[chan];
    auto pedBaseLine = mSimCondition->channels[chan].pedestal;
    for (int ib = NTimeBinsPerBC; ib--;) {
      bcdata[chan][ib] += gRandom->Gaus(pedBaseLine + mPedestalBLFluct[chan], chanConf.pedestalNoise);
      int adc = std::nearbyint(bcdata[chan][ib]);
      bcdata[chan][ib] = adc < ADCMax ? (adc > ADCMin ? adc : ADCMin) : ADCMax;
    }
  }
  bc.digitized = true;
}

bool Digitizer::triggerBC(int ibc)
{
  // check trigger for the cached BC in the position ibc
  auto& bcCached = *mFastCache[ibc];

  LOG(DEBUG) << "CHECK TRIGGER " << ibc << " IR=" << bcCached;
  if (mIsContinuous) {
    for (int ic = mTriggerConfig.size(); ic--;) {
      const auto& trigCh = mTriggerConfig[ic];
      bool okPrev = false;
      int last1 = trigCh.last + 2;
      // look for 2 consecutive bins (the 1st one spanning trigCh.first : trigCh.last range) so that
      // signal[bin]-signal[bin+trigCh.shift] > trigCh.threshold
      for (int ib = trigCh.first; ib < last1; ib++) { // ib may be negative, so we shift by offs and look in the ADC cache
        int binF, bcFidx = ibc + binHelper(ib, binF);
        int binL, bcLidx = ibc + binHelper(ib + trigCh.shift, binL);
        const auto& bcF = (bcFidx < 0 || !mFastCache[bcFidx]) ? mDummyBC : *mFastCache[bcFidx];
        const auto& bcL = (bcLidx < 0 || !mFastCache[bcLidx]) ? mDummyBC : *mFastCache[bcLidx];
        bool ok = bcF.data[trigCh.id][binF] - bcL.data[trigCh.id][binL] > trigCh.threshold;
        if (ok && okPrev) {                          // trigger ok!
          bcCached.trigChanMask |= 0x1 << trigCh.id; // register trigger mask
          LOG(DEBUG) << " triggering channel " << int(trigCh.id) << " => " << bcCached.trigChanMask;
          break;
        }
        okPrev = ok;
      }
    } // loop over trigger channels
  } else {
    // just check if this BC IR corresponds to externall trigger
    if (!mIRExternalTrigger.empty() && mIRExternalTrigger.front() == bcCached) {
      bcCached.trigChanMask = AllChannelsMask;
      mIRExternalTrigger.pop_front(); // suppress accounted external trigger
    }
  }

  if (bcCached.trigChanMask) { // there are triggered channels, flag modules/channels to read
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
  for (int ic = 0; ic < NChannels; ic++) {
    if (chan2Store & (0x1 << ic)) {
      digitsCh.emplace_back(ic, bc.data[ic]);
      nSto++;
    }
  }
  int nBC = digitsBC.size();
  digitsBC.emplace_back(first, nSto, bc, chan2Store, bc.trigChanMask);
  // TODO clarify if we want to store MC labels for all channels or only for stored ones
  for (const auto& lbl : bc.labels) {
    if (chan2Store & (0x1 << lbl.getChannel())) {
      labels.addElement(nBC, lbl);
    }
  }
}

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
        auto signal = chanConfig.shape[sample] * nphe; // signal accounting for the gain
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
}

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
      if (md.id >= 0) {
        mModConfAux.emplace_back(md);
        //
        for (int ic = Module::MaxChannels; ic--;) {
          if (md.trigChannel[ic]) { // check if this triggering channel was already registered
            bool skip = false;
            for (int is = mTriggerConfig.size(); is--;) {
              if (mTriggerConfig[is].id == md.channelID[ic]) {
                skip = true;
                break;
              }
            }
            if (!skip) {
              const auto& trgChanConf = md.trigChannelConf[ic];
              if (trgChanConf.last + trgChanConf.shift + 1 >= NTimeBinsPerBC) {
                LOG(FATAL) << "Wrong trigger settings";
              }
              mTriggerConfig.emplace_back(trgChanConf);
              LOG(INFO) << "Adding channel " << int(trgChanConf.id) << '(' << channelName(trgChanConf.id) << ") as triggering one";
              if (trgChanConf.first < mTrigBinMin) {
                mTrigBinMin = trgChanConf.first;
              }
              if (trgChanConf.last + trgChanConf.shift > mTrigBinMax) {
                mTrigBinMax = trgChanConf.last + trgChanConf.shift;
              }
            }
          }
        }
      }
    }
    if (int(mTriggerConfig.size()) > MaxTriggerChannels) {
      LOG(FATAL) << "Too many triggering channels (" << mTriggerConfig.size() << ')';
    }
    mModuleConfig->print();
    //
  }

  if (!mSimCondition) { // load this only once
    mSimCondition = mgr.get<SimCondition>(CCDBPathConfigSim);
    LOG(INFO) << "Loaded simulation configuration for timestamp " << mTimeStamp;
    mSimCondition->print();
  }
}

//______________________________________________________________
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
