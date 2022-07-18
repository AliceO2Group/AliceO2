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
#include "ZDCSimulation/DigitizerTest.h"
#include <TRandom.h>
#include <TMath.h>

using namespace o2::zdc;

//______________________________________________________________________________
void DigitizerTest::init()
{
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  if (mCCDBServer.size() > 0) {
    mgr.setURL(mCCDBServer);
  }
  LOG(info) << "Initialization of ZDC Test Digitization " << mgr.getURL();

  mSimCondition = mgr.get<o2::zdc::SimCondition>(o2::zdc::CCDBPathConfigSim);
  if (!mSimCondition) {
    LOG(fatal) << "Missing SimCondition configuration object @ " << o2::zdc::CCDBPathConfigSim;
    return;
  }
  LOG(info) << "Loaded SimCondition for timestamp " << mgr.getTimestamp();
  mSimCondition->print();

  mModuleConfig = mgr.get<o2::zdc::ModuleConfig>(o2::zdc::CCDBPathConfigModule);
  if (!mModuleConfig) {
    LOG(fatal) << "Missing ModuleConfig configuration object @ " << o2::zdc::CCDBPathConfigModule;
    return;
  }
  LOG(info) << "Loaded ModuleConfig for timestamp " << mgr.getTimestamp();
  mModuleConfig->print();
}

//______________________________________________________________________________
void DigitizerTest::setMask(uint32_t ich, uint32_t mask)
{
  if (ich < NChannels) {
    mMask[ich] = mask;
  } else {
    LOG(fatal) << "Setting mask for non existing channel " << ich;
  }
}

o2::zdc::Digitizer::BCCache& DigitizerTest::getCreateBCCache(const o2::InteractionRecord& ir)
{
  // printf("size = %zu Add %u.%u\n", mCache.size(), ir.orbit, ir.bc);
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
double DigitizerTest::add(int ic, float myAmp, const o2::InteractionRecord irpk,
                          float myShift, bool hasJitter)
{
  // myShift is the position of the signal peak inside the bunch crossing identified by irpk (in ns)
  // and is converted into sample units
  myShift = myShift * NTimeBinsPerBC / o2::constants::lhc::LHCBunchSpacingNS;
  // Shift of samples due to additional fluctuations
  float shift = hasJitter ? gRandom->Gaus(0, mSimCondition->channels[ic].pedestalNoise) : 0;
  int pos_min = mSimCondition->channels[ic].ampMinID;
  int nbx = mSimCondition->channels[ic].shape.size();
  // This may result into a wrap of the BC counter. However the signal is stored only
  // in pre-allocated containers
  o2::InteractionRecord ir = irpk + BCCacheMin;
  for (Int_t ib = 0; ib < NBC2Cache; ib++) {
    auto* bc = getBCCache(ir);
    if (bc) {
      // Flag the presence of channel data
      auto& bcd = getCreateBCData(ir);
      bcd.channels = bcd.channels | mMask[ic];
      auto sbcd = NTimeBinsPerBC * ir.differenceInBC(irpk);
      int sample = TMath::Nint(pos_min + TSN * (sbcd - myShift) + shift + TSNH);
      for (int i = 0; i < NTimeBinsPerBC; i++) {
        if (sample >= 0 && sample < nbx) {
          double y = mSimCondition->channels[ic].shape[sample];
          bc->data[ic][i] += y * myAmp;
          // LOG(info) << ic << " " << ir.orbit << "." << ir.bc << " s" << i << ") s " << sample << " " << y*myAmp << " -> " << bc->data[ic][i];
        }
        sample += TSN;
      }
    }
    ir++;
  }
  // Assigned position in units of bunch crossing
  return irpk.orbit * o2::constants::lhc::LHCMaxBunches + irpk.bc + (myShift + shift / double(TSN)) / double(NTimeBinsPerBC);
} // add

//______________________________________________________________________________
void DigitizerTest::digitize()
{
  float nba = mModuleConfig->nBunchAverage;
  for (auto bc = mCache.begin(); bc != mCache.end(); bc++) {
    if (zdcOrbitData.empty() || bc->orbit != zdcOrbitData.back().ir.orbit) {
      auto& od = zdcOrbitData.emplace_back();
      od.ir.orbit = bc->orbit;
      od.ir.bc = o2::constants::lhc::LHCMaxBunches - 1;
      // Rough estimate of pedestal fluctuations to fill orbit data
      for (int ic = 0; ic < NChannels; ic++) {
        auto base_m = mSimCondition->channels[ic].pedestal;      // Average pedestal
        auto base_s = mSimCondition->channels[ic].pedestalFluct; // Baseline oscillations
        auto base_n = mSimCondition->channels[ic].pedestalNoise; // Electronic noise
        // We don't know the time scale of the fluctuations of the baseline. As a
        // rough guess we consider two bunch crossings
        // sum = 12 * (mNEmptyBCs/2) * (2*base_m) = 12 * mNEmptyBCs * base_m
        float mean_sum = 12. * nba * base_m;                     // Adding 12 samples for bunch crossing
        float rms_sum = 12. * 2. * base_s * std::sqrt(nba / 2.); // 2 for fluctuation every 2 BCs
        float rms_noise_sum = base_n * std::sqrt(12. * nba);
        float ped = gRandom->Gaus(mean_sum, rms_sum) + gRandom->Gaus(0, rms_noise_sum);
        int16_t peds = std::round(ped / nba / 12. / mModuleConfig->baselineFactor);
        if (peds < SHRT_MIN) {
          peds = SHRT_MIN;
        } else if (peds > SHRT_MAX) {
          peds = SHRT_MAX;
        }
        od.data[ic] = peds;
      }
      // printf("Adding data for orbit=%u\n", od.ir.orbit);
    }
    // printf("Adding CH data for bunch=%u\n", bc->bc);
    auto& bcd = getCreateBCData(*bc);
    bcd.ref.setFirstEntry(zdcChData.size());
    uint32_t nstored = 0;
    for (int ic = 0; ic < NChannels; ic++) {
      // Check if channel has data for this bunch crossing
      if (bcd.channels & mMask[ic]) {
        Double_t meanp = mSimCondition->channels[ic].pedestal;
        Double_t sigmab = mSimCondition->channels[ic].pedestalFluct;
        // LOG(info) << ic << " meanp=" << meanp << " sigmab=" << sigmab;
        //  No pedestal oscillations on data. We do it in the orbit data
        for (int i = 0; i < NTimeBinsPerBC; i++) {
          Double_t yval = TMath::Nint(bc->data[ic][i] + gRandom->Gaus(meanp, sigmab));
          if (yval > ADCMax) {
            yval = ADCMax;
          } else if (yval < ADCMin) {
            yval = ADCMin;
          }
          // This is not correct but will work for direct digitization
          bc->digi[ic][i] = yval;
          // LOG(info) << ic << " " << i << ") " << yval;
        }
        // Store data in output array
        zdcChData.emplace_back(ic, bc->digi[ic]);
        nstored++;
      }
    }
    bcd.ref.setEntries(nstored);
  }
} // digitize

//______________________________________________________________________________
void DigitizerTest::clear()
{
  mCache.clear();
  zdcOrbitData.clear();
  zdcBCData.clear();
  zdcChData.clear();
}

//______________________________________________________________________________
BCData& DigitizerTest::getCreateBCData(const o2::InteractionRecord& ir)
{
  if (zdcBCData.empty() || zdcBCData.back().ir < ir) {
    zdcBCData.emplace_back();
    auto& bc = zdcBCData.back();
    bc.ir = ir;
    return bc;
  }
  if (zdcBCData.front().ir > ir) {
    zdcBCData.emplace(zdcBCData.begin());
    auto& cb = zdcBCData.front();
    cb.ir = ir;
    return cb;
  }
  for (auto cb = zdcBCData.begin(); cb != zdcBCData.end(); cb++) {
    if ((*cb).ir == ir) {
      return *cb;
    }
    if (ir < (*cb).ir) {
      auto cbnew = zdcBCData.emplace(cb); // insert new element before cb
      (*cbnew).ir = ir;
      return (*cbnew);
    }
  }
  return zdcBCData.front();
}

//______________________________________________________________________________
o2::zdc::Digitizer::BCCache* DigitizerTest::getBCCache(const o2::InteractionRecord& ir)
{
  // get pointer on existing cache
  for (auto cb = mCache.begin(); cb != mCache.end(); cb++) {
    if ((*cb) == ir) {
      return &(*cb);
    }
  }
  return nullptr;
}
