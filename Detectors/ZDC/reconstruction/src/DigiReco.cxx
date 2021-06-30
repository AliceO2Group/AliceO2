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

#include <TMath.h>
#include "Framework/Logger.h"
#include "ZDCReconstruction/DigiReco.h"
#include "ZDCReconstruction/RecoParamZDC.h"

namespace o2
{
namespace zdc
{
using O2_ZDC_DIGIRECO_FLT = float;

void DigiReco::init()
{
  LOG(INFO) << "Initialization of ZDC reconstruction";
  // Load configuration parameters
  auto& sopt = ZDCSimParam::Instance();
  mIsContinuous = sopt.continuous;
  mNBCAHead = mIsContinuous ? sopt.nBCAheadCont : sopt.nBCAheadTrig;

  if (!mModuleConfig) {
    LOG(FATAL) << "Missing ModuleConfig configuration object";
    return;
  }

  // Prepare tapered sinc function
  // tsc/TSN =3.75 (~ 4) and TSL*TSN*sqrt(2)/tsc >> 1 (n. of sigma)
  const O2_ZDC_DIGIRECO_FLT tsc = 750;
  int n = TSL * TSN;
  for (int tsi = 0; tsi <= n; tsi++) {
    O2_ZDC_DIGIRECO_FLT arg1 = TMath::Pi() * O2_ZDC_DIGIRECO_FLT(tsi) / O2_ZDC_DIGIRECO_FLT(TSN);
    O2_ZDC_DIGIRECO_FLT fs = 1;
    if (arg1 != 0) {
      fs = TMath::Sin(arg1) / arg1;
    }
    O2_ZDC_DIGIRECO_FLT arg2 = O2_ZDC_DIGIRECO_FLT(tsi) / tsc;
    O2_ZDC_DIGIRECO_FLT fg = TMath::Exp(-arg2 * arg2);
    mTS[n + tsi] = fs * fg;
    mTS[n - tsi] = mTS[n + tsi]; // Function is even
  }

  if (mTreeDbg) {
    // Open debug file
    LOG(INFO) << "ZDC DigiReco: opening debug output";
    mDbg = std::make_unique<TFile>("ZDCRecoDbg.root", "recreate");
    mTDbg = std::make_unique<TTree>("zdcr", "ZDCReco");
    mTDbg->Branch("zdcr", "RecEventAux", &mRec);
  }

  // Update reconstruction parameters
  //auto& ropt=RecoParamZDC::Instance();
  o2::zdc::RecoParamZDC& ropt = const_cast<o2::zdc::RecoParamZDC&>(RecoParamZDC::Instance());

  // Fill maps to decode the pattern of channels with hit
  for (int itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    // If the reconstruction parameters were not manually set
    if (ropt.tmod[itdc] < 0 || ropt.tch[itdc] < 0) {
      int isig = TDCSignal[itdc];
      for (int im = 0; im < NModules; im++) {
        for (uint32_t ic = 0; ic < NChPerModule; ic++) {
          if (mModuleConfig->modules[im].channelID[ic] == isig && mModuleConfig->modules[im].readChannel[ic]) {
            //ropt.updateFromString(TString::Format("RecoParamZDC.tmod[%d]=%d;",itdc,im));
            //ropt.updateFromString(TString::Format("RecoParamZDC.tch[%d]=%d;",itdc,ic));
            ropt.tmod[itdc] = im;
            ropt.tch[itdc] = ic;
            // Fill mask to identify TDC channels
            mTDCMask[itdc] = (0x1 << (4 * im + ic));
            goto next_itdc;
          }
        }
      }
    }
  next_itdc:;
    LOG(INFO) << "TDC " << itdc << "(" << ChannelNames[TDCSignal[itdc]] << ")"
              << " mod " << ropt.tmod[itdc] << " ch " << ropt.tch[itdc];
  }

  // TDC calibration
  for (int itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    float fval = ropt.tdc_shift[itdc];
    // If the reconstruction parameters were not manually set
    if (fval < 0) {
      // Check if calibration object is present
      if (!mTDCParam) {
        LOG(FATAL) << "TDC " << itdc << " missing configuration object and no manual override";
      } else {
        fval = mTDCParam->getShift(itdc) / FTDCVal;
      }
    }
    auto val = std::nearbyint(fval);
    if (val < kMinShort) {
      LOG(FATAL) << "Shift for TDC " << itdc << " " << val << " is out of range";
    }
    if (val > kMaxShort) {
      LOG(FATAL) << "Shift for TDC " << itdc << " " << val << " is out of range";
    }
    tdc_shift[itdc] = val;
    LOG(INFO) << itdc << " " << ChannelNames[TDCSignal[itdc]] << " shift= " << tdc_shift[itdc] << " i.s. = " << val * FTDCVal << " ns";
  }

  // TDC search zone
  for (int itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    // If the reconstruction parameters were not manually set
    if (ropt.tdc_search[itdc] <= 0) {
      if (!mRecoConfigZDC) {
        LOG(FATAL) << "Search zone for TDC " << itdc << " missing configuration object and no manual override";
      } else {
        ropt.tdc_search[itdc] = mRecoConfigZDC->tdc_search[itdc];
      }
    }
    LOG(INFO) << itdc << " " << ChannelNames[TDCSignal[itdc]] << " search= " << ropt.tdc_search[itdc] << " i.s. = " << ropt.tdc_search[itdc] * FTDCVal << " ns";
  }

  // Energy calibration
  for (int il = 0; il < ChEnergyCalib.size(); il++) {
    if (ropt.energy_calib[ChEnergyCalib[il]] > 0) {
      LOG(INFO) << "Energy Calibration from command line " << ChannelNames[ChEnergyCalib[il]] << " = " << ropt.energy_calib[ChEnergyCalib[il]];
    } else if (mEnergyParam && mEnergyParam->energy_calib[ChEnergyCalib[il]] > 0) {
      ropt.energy_calib[ChEnergyCalib[il]] = mEnergyParam->energy_calib[ChEnergyCalib[il]];
      LOG(INFO) << "Energy Calibration from CCDB " << ChannelNames[ChEnergyCalib[il]] << " = " << ropt.energy_calib[ChEnergyCalib[il]];
    } else {
      if (ChEnergyCalib[il] == CaloCommonPM[ChEnergyCalib[il]]) {
        // Is a common PM or a ZEM
        ropt.energy_calib[ChEnergyCalib[il]] = 1;
        LOG(WARNING) << "Default Energy Calibration  " << ChannelNames[ChEnergyCalib[il]] << " = " << ropt.energy_calib[ChEnergyCalib[il]];
      } else {
        // Is one of the analog sums -> same calibration as common PM
        // N.B. the calibration for common has already been set in the loop
        ropt.energy_calib[ChEnergyCalib[il]] = ropt.energy_calib[CaloCommonPM[il]];
        LOG(INFO) << "SUM Energy Calibration  " << ChannelNames[ChEnergyCalib[il]] << " = " << ropt.energy_calib[ChEnergyCalib[il]];
      }
    }
  }

  // Tower calibration
  for (int il = 0; il < ChTowerCalib.size(); il++) {
    if (ropt.tower_calib[ChTowerCalib[il]] > 0) {
      LOG(INFO) << "Tower Calibration from command line " << ChannelNames[ChTowerCalib[il]] << " = " << ropt.tower_calib[ChTowerCalib[il]];
    } else if (mTowerParam && mTowerParam->tower_calib[ChTowerCalib[il]] > 0) {
      ropt.tower_calib[ChTowerCalib[il]] = mTowerParam->tower_calib[ChTowerCalib[il]];
      LOG(INFO) << "Tower Calibration from CCDB " << ChannelNames[ChTowerCalib[il]] << " = " << ropt.tower_calib[ChTowerCalib[il]];
    } else {
      ropt.tower_calib[ChTowerCalib[il]] = 1;
      LOG(WARNING) << "Default Tower Calibration  " << ChannelNames[ChTowerCalib[il]] << " = " << ropt.tower_calib[ChTowerCalib[il]];
    }
  }

  // Tower energy calibration
  for (int il = 0; il < ChTowerCalib.size(); il++) {
    if (ropt.energy_calib[ChTowerCalib[il]] > 0) {
      LOG(INFO) << "Tower Energy Calibration from command line " << ChannelNames[ChTowerCalib[il]] << " = " << ropt.energy_calib[ChTowerCalib[il]];
    } else {
      ropt.energy_calib[ChTowerCalib[il]] = ropt.tower_calib[ChTowerCalib[il]] * ropt.energy_calib[CaloCommonPM[ChTowerCalib[il]]];
      LOG(INFO) << "Tower Energy Calibration " << ChannelNames[ChTowerCalib[il]] << " = " << ropt.energy_calib[ChTowerCalib[il]];
    }
  }

  // Fill maps channel maps for integration
  for (int ich = 0; ich < NChannels; ich++) {
    // If the reconstruction parameters were not manually set
    if (ropt.amod[ich] < 0 || ropt.ach[ich] < 0) {
      for (int im = 0; im < NModules; im++) {
        for (uint32_t ic = 0; ic < NChPerModule; ic++) {
          if (mModuleConfig->modules[im].channelID[ic] == ich && mModuleConfig->modules[im].readChannel[ic]) {
            ropt.amod[ich] = im;
            ropt.ach[ich] = ic;
            goto next_ich;
          }
        }
      }
    }
  next_ich:;
    LOG(INFO) << "ADC " << ich << "(" << ChannelNames[ich] << ") mod " << ropt.amod[ich] << " ch " << ropt.ach[ich];
  }

  // Integration ranges
  for (int ich = 0; ich < NChannels; ich++) {
    // If the reconstruction parameters were not manually set
    if (ropt.beg_int[ich] == DummyIntRange || ropt.end_int[ich] == DummyIntRange) {
      if (!mRecoConfigZDC) {
        LOG(FATAL) << "Integration for signal " << ich << " missing configuration object and no manual override";
      } else {
        ropt.beg_int[ich] = mRecoConfigZDC->beg_int[ich];
        ropt.end_int[ich] = mRecoConfigZDC->end_int[ich];
      }
    }
    if (ropt.beg_ped_int[ich] == DummyIntRange || ropt.end_ped_int[ich] == DummyIntRange) {
      if (!mRecoConfigZDC) {
        LOG(ERROR) << "Integration for pedestal " << ich << " missing configuration object and no manual override";
      } else {
        ropt.beg_ped_int[ich] = mRecoConfigZDC->beg_ped_int[ich];
        ropt.end_ped_int[ich] = mRecoConfigZDC->end_ped_int[ich];
      }
    }
    LOG(INFO) << ChannelNames[ich] << " integration: signal=[" << ropt.beg_int[ich] << ":" << ropt.end_int[ich] << "] pedestal=[" << ropt.beg_ped_int[ich] << ":" << ropt.end_ped_int[ich] << "]";
  }
}

int DigiReco::process(const gsl::span<const o2::zdc::OrbitData>& orbitdata, const gsl::span<const o2::zdc::BCData>& bcdata, const gsl::span<const o2::zdc::ChannelData>& chdata)
{
  // We assume that vectors contain data from a full time frame
  mOrbitData = orbitdata;
  mBCData = bcdata;
  mChData = chdata;

  // Initialization of lookup structure for pedestals
  mOrbit.clear();
  int norb = mOrbitData.size();
  for (int iorb = 0; iorb < norb; iorb++) {
    mOrbit[mOrbitData[iorb].ir.orbit] = iorb;
    if (mVerbosity >= DbgFull) {
      LOG(INFO) << "mOrbitData[" << mOrbitData[iorb].ir.orbit << "] = " << iorb;
    }
  }
  mNBC = mBCData.size();
  mReco.clear();
  mReco.resize(mNBC);
  // Initialization of reco structure
  for (int ibc = 0; ibc < mNBC; ibc++) {
    auto& bcr = mReco[ibc];
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
      for (int i = 0; i < MaxTDCValues; i++) {
        bcr.tdcVal[itdc][i] = kMinShort;
        bcr.tdcAmp[itdc][i] = kMinShort;
      }
    }
    auto& bcd = mBCData[ibc];
    bcr.ir = bcd.ir;
    int chEnt = bcd.ref.getFirstEntry();
    for (int ic = 0; ic < bcd.ref.getEntries(); ic++) {
      auto& chd = mChData[chEnt];
      if (chd.id > IdDummy && chd.id < NChannels) {
        bcr.ref[chd.id] = chEnt;
      }
      chEnt++;
    }
  }

  // Probably this is not necessary
  //   for(int itdc=0; itdc<NTDCChannels; itdc++){
  //     mReco.pattern[itdc]=0;
  //     for(int itb=0; itb<NTimeBinsPerBC; itb++){
  //       mReco.fired[itdc][itb]=0;
  //     }
  //     for(int isb=0; isb<mNSB; isb++){
  //       mReco.inter[itdc][isb]=0;
  //     }
  //   }

  // Assign interaction record and event information
  for (int ibc = 0; ibc < mNBC; ibc++) {
    mReco[ibc].ir = mBCData[ibc].ir;
    mReco[ibc].channels = mBCData[ibc].channels;
    mReco[ibc].triggers = mBCData[ibc].triggers;
  }

  // Find consecutive bunch crossings and perform signal interpolation
  // in the identified ranges (in the reconstruction method we take into
  // account for signals that do not span the entire reange)
  int seq_beg = 0;
  int seq_end = 0;
  LOG(INFO) << "ZDC reconstruction for " << mNBC << " bunch crossings";
  for (int ibc = 0; ibc < mNBC; ibc++) {
    auto& ir = mBCData[seq_end].ir;
    auto bcd = mBCData[ibc].ir.differenceInBC(ir);
    if (bcd < 0) {
      LOG(FATAL) << "Orbit number is not increasing " << mBCData[seq_end].ir.orbit << "." << mBCData[seq_end].ir.bc << " followed by " << mBCData[ibc].ir.orbit << "." << mBCData[ibc].ir.bc;
      return __LINE__;
    } else if (bcd > 1) {
      // Detected a gap
      reconstruct(seq_beg, seq_end);
      seq_beg = ibc;
      seq_end = ibc;
    } else if (ibc == (mNBC - 1)) {
      // Last bunch
      seq_end = ibc;
      reconstruct(seq_beg, seq_end);
      seq_beg = mNBC;
      seq_end = mNBC;
    } else {
      // Look for another bunch
      seq_end = ibc;
    }
  }

  return 0;
}

int DigiReco::reconstruct(int ibeg, int iend)
{
  // Process consecutive BCs
  if (ibeg == iend) {
    if (mReco[ibeg].ir.bc == (o2::constants::lhc::LHCMaxBunches - 1)) {
      mNLastLonely++;
    } else {
      mNLonely++;
      LOG(INFO) << "Lonely bunch " << mReco[ibeg].ir.orbit << "." << mReco[ibeg].ir.bc;
    }
    return 0;
  }

  if (mVerbosity >= DbgFull) {
    LOG(INFO) << __func__ << "(" << ibeg << "," << iend << "): " << mReco[ibeg].ir.orbit << "." << mReco[ibeg].ir.bc << " - " << mReco[iend].ir.orbit << "." << mReco[iend].ir.bc;
  }

  // Get reconstruction parameters
  auto& ropt = RecoParamZDC::Instance();

  // Apply differential discrimination with triple condition
  for (int itdc = 0; itdc < NTDCChannels; itdc++) {
    // Check if channel has valid data for consecutive bunches in current bunch range
    // N.B. there are events recorded from ibeg-iend but we are not sure if it is the
    // case for every TDC channel
    int istart = -1, istop = -1;
    // Loop allows for gaps in the data sequence for each TDC channel
    for (int ibun = ibeg; ibun <= iend; ibun++) {
      if (mBCData[ibun].channels & mTDCMask[itdc]) { // TDC channel has data for this event
        if (istart < 0) {
          istart = ibun;
        }
        istop = ibun;
      } else { // No data from channel
        // A gap is detected
        if (istart >= 0 && (istop - istart) > 0) {
          // Need data for at least two consecutive bunch crossings
          processTrigger(itdc, istart, istop);
        }
        istart = -1;
        istop = -1;
      }
    }
    // Check if there are consecutive bunch crossings at the end of group
    if (istart >= 0 && (istop - istart) > 0) {
      processTrigger(itdc, istart, istop);
    }
  }

  // Loop on bunches after trigger evaluation
  // Reconstruct integrated charges and fill output tree
  // TODO: compare average pedestal with estimation from current event
  // TODO: failover in case of discrepancy
  for (int ibun = ibeg; ibun <= iend; ibun++) {
    // Look for offset
    float pbun[NChannels];
    auto orbit = mBCData[ibun].ir.orbit;
    std::map<uint32_t, int>::iterator it = mOrbit.find(orbit);
    if (it != mOrbit.end()) {
      // Subtract pedestal
      auto& orbitdata = mOrbitData[it->second];
      for (int ich = 0; ich < NChannels; ich++) {
        pbun[ich] = orbitdata.asFloat(ich);
      }
    } else {
      LOG(ERROR) << "Missing pedestal for bunch " << ibun;
      for (int ich = 0; ich < NChannels; ich++) {
        pbun[ich] = std::numeric_limits<float>::infinity();
      }
    }

#ifdef O2_ZDC_DEBUG
    // Debug dump of pedestal
    for (int ich = 0; ich < NChannels; ich++) {
      LOG(INFO) << "bunch: " << ibun << " ch: " << ich << " " << ChannelNames[ich] << " offset: " << pbun[ich];
    }
#endif

    auto& rec = mReco[ibun];
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
#ifdef O2_ZDC_DEBUG
      if (rec.fired[itdc] != 0x0) {
        printf("%d %u.%u TDC %d %x", ibun, rec.ir.orbit, rec.ir.bc, itdc, rec.fired[itdc]);
        for (int isam = 0; isam < NTimeBinsPerBC; isam++) {
          printf("%d", rec.fired[itdc] & mMask[isam] ? 1 : 0);
        }
        printf("\n");
      }
#endif
      rec.pattern[itdc] = 0;
      for (int32_t i = 0; i < rec.ntdc[itdc]; i++) {
#ifdef O2_ZDC_DEBUG
        LOG(INFO) << "tdc " << i << " [" << ChannelNames[TDCSignal[itdc]] << "] " << rec.tdcAmp[itdc][i] << " @ " << rec.tdcVal[itdc][i];
#endif
        // There is a TDC value in the search zone around main-main position
        if (std::abs(rec.tdcVal[itdc][i]) < ropt.tdc_search[itdc]) {
          rec.pattern[itdc] = 1;
        }
#ifdef O2_ZDC_DEBUG
        else {
          LOG(INFO) << rec.tdcVal[itdc][i] << " " << ropt.tdc_search[itdc];
        }
#endif
      }
    }
#ifdef O2_ZDC_DEBUG
    printf("%d %u.%-4u TDC PATTERN: ", ibun, mReco[ibun].ir.orbit, mReco[ibun].ir.bc);
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
      printf("%d", rec.pattern[itdc]);
    }
    printf("\n");
#endif
    // Check if coincidence of common PM and sum of towers is satisfied
    bool fired[NChannels] = {0};
    // Side A
    if ((rec.pattern[TDCZNAC] || ropt.bitset[TDCZNAC]) && (rec.pattern[TDCZNAS] || ropt.bitset[TDCZNAS])) {
      for (int ich = IdZNAC; ich <= IdZNASum; ich++) {
        fired[ich] = true;
      }
    }
    if ((rec.pattern[TDCZPAC] || ropt.bitset[TDCZPAC]) && (rec.pattern[TDCZPAS] || ropt.bitset[TDCZPAS])) {
      for (int ich = IdZPAC; ich <= IdZPASum; ich++) {
        fired[ich] = true;
      }
    }
    // ZEM1 and ZEM2 are not in coincidence
    fired[IdZEM1] = rec.pattern[TDCZEM1];
    fired[IdZEM2] = rec.pattern[TDCZEM2];
    // Side C
    if ((rec.pattern[TDCZNCC] || ropt.bitset[TDCZNCC]) && (rec.pattern[TDCZNCS] || ropt.bitset[TDCZNCS])) {
      for (int ich = IdZNCC; ich <= IdZNCSum; ich++) {
        fired[ich] = true;
      }
    }
    if ((rec.pattern[TDCZPCC] || ropt.bitset[TDCZPCC]) && (rec.pattern[TDCZPCS] || ropt.bitset[TDCZPCS])) {
      for (int ich = IdZPCC; ich <= IdZPCSum; ich++) {
        fired[ich] = true;
      }
    }

    if (mVerbosity >= DbgFull) {
      printf("%d FIRED ", ibun);
      printf("ZNA:%d%d%d%d%d%d ZPA:%d%d%d%d%d%d ZEM:%d%d ZNC:%d%d%d%d%d%d ZPC:%d%d%d%d%d%d\n",
             fired[IdZNAC], fired[IdZNA1], fired[IdZNA2], fired[IdZNA3], fired[IdZNA4], fired[IdZNASum],
             fired[IdZPAC], fired[IdZPA1], fired[IdZPA2], fired[IdZPA3], fired[IdZPA4], fired[IdZPASum],
             fired[IdZEM1], fired[IdZEM2],
             fired[IdZNCC], fired[IdZNC1], fired[IdZNC2], fired[IdZNC3], fired[IdZNC4], fired[IdZNCSum],
             fired[IdZPCC], fired[IdZPC1], fired[IdZPC2], fired[IdZPC3], fired[IdZPC4], fired[IdZPCSum]);
    }
    for (int ich = 0; ich < NChannels; ich++) {
      // Check if the corresponding TDC is fired
      if (fired[ich]) {
        // Check if channel data are present in payload
        auto ref = mReco[ibun].ref[ich];
        if (ref < ZDCRefInitVal) {
          float sum = 0;
          for (int is = ropt.beg_int[ich]; is <= ropt.end_int[ich]; is++) {
            // TODO: fallback if offset is missing
            // TODO: fallback if channel has pile-up
            // TODO: manage signal positioned across boundary
            sum += (pbun[ich] - float(mChData[ref].data[is]));
          }
#ifdef O2_ZDC_DEBUG
          printf("CH %2d %s: %f\n", ich, ChannelNames[ich].data(), sum);
#endif
          rec.ezdc[ich] = sum * ropt.energy_calib[ich];
        }
      }
    }
    if (mTreeDbg) {
      mRec = rec;
      mTDbg->Fill();
    }
  } // Loop on bunches
  return 0;
}

void DigiReco::processTrigger(int itdc, int ibeg, int iend)
{
#ifdef O2_ZDC_DEBUG
  LOG(INFO) << __func__ << "(itdc=" << itdc << "[" << ChannelNames[TDCSignal[itdc]] << "] ," << ibeg << "," << iend << "): " << mReco[ibeg].ir.orbit << "." << mReco[ibeg].ir.bc << " - " << mReco[iend].ir.orbit << "." << mReco[iend].ir.bc;
#endif
  // Get reconstruction parameters
  auto& ropt = RecoParamZDC::Instance();

  int nbun = iend - ibeg + 1;
  int maxs2 = NTimeBinsPerBC * nbun - 1;
  int shift = ropt.tsh[itdc];
  int thr = ropt.tth[itdc];

  int is1 = 0, is2 = 1;
  int isfired[3] = {0};
  int it1 = 0, it2 = 0, ib1 = -1, ib2 = -1;
  for (;;) {
    // Shift data
    for (int i = 1; i < 3; i++) {
      isfired[i] = isfired[i - 1];
    }
    // Bunches and samples that are used in the difference
    int b1 = ibeg + is1 / NTimeBinsPerBC;
    int b2 = ibeg + is2 / NTimeBinsPerBC;
    int s1 = is1 % NTimeBinsPerBC;
    int s2 = is2 % NTimeBinsPerBC;
    auto ref_m = mReco[b1].ref[TDCSignal[itdc]];
    auto ref_s = mReco[b2].ref[TDCSignal[itdc]];
    // Check data consistency before computing difference
    if (ref_m == ZDCRefInitVal || ref_s == ZDCRefInitVal) {
      LOG(FATAL) << "Missing information for bunch crossing";
      return;
    }
    // TODO: More checks that bunch crossings are indeed consecutive
    int diff = mChData[ref_m].data[s1] - mChData[ref_s].data[s2];
    // Triple trigger condition
    if (diff > thr) {
      isfired[0] = 1;
      if (isfired[1] == 1 && isfired[2] == 1) {
        // Fired bit is assigned to the second sample, i.e. to the one that can identify the
        // signal peak position
        mReco[b2].fired[itdc] |= mMask[s2];
#ifdef O2_ZDC_DEBUG
        LOG(INFO) << itdc << " " << ChannelNames[TDCSignal[itdc]] << " Fired @ " << mReco[b2].ir.orbit << "." << mReco[b2].ir.bc << ".s" << s2;
#endif
      }
    }
    if (is2 >= shift) {
      is1++;
    }
    if (is2 < maxs2) {
      is2++;
    }
    if (is1 == maxs2) {
      break;
    }
  }
  interpolate(itdc, ibeg, iend);
}

void DigiReco::interpolate(int itdc, int ibeg, int iend)
{
#ifdef O2_ZDC_DEBUG
  LOG(INFO) << __func__ << "(itdc=" << itdc << "[" << ChannelNames[TDCSignal[itdc]] << "] ," << ibeg << "," << iend << "): " << mReco[ibeg].ir.orbit << "." << mReco[ibeg].ir.bc << " - " << mReco[iend].ir.orbit << "." << mReco[iend].ir.bc;
#endif
  // TODO: get data from preceding time frame
  constexpr int MaxTimeBin = NTimeBinsPerBC - 1; //< number of samples per BC
  constexpr int tsnh = TSN / 2;                  // Half number of points in interpolation
  constexpr int nsbun = TSN * NTimeBinsPerBC;    // Total number of interpolated points per bunch crossing
  int nbun = iend - ibeg + 1;                    // Number of adjacent bunches
  int nsam = nbun * NTimeBinsPerBC;              // Number of acquired samples
  int ntot = nsam * TSN;                         // Total number of points in the interpolated arrays
  int nint = (nbun * NTimeBinsPerBC - 1) * TSN;  // Total points in the interpolation region (-1)
  constexpr int nsp = 5;                         // Number of points to be searched

  // At this level there should be no need to check if the TDC channel is connected
  // since a fatal should have been raised already
  for (int ibun = ibeg; ibun <= iend; ibun++) {
    auto ref = mReco[ibun].ref[TDCSignal[itdc]];
    if (ref == ZDCRefInitVal) {
      LOG(FATAL) << "Missing information for bunch crossing";
    }
  }

  // Get reconstruction parameters
  auto& ropt = RecoParamZDC::Instance();

  int imod = ropt.tmod[itdc]; // Module corresponding to TDC channel
  int ich = ropt.tch[itdc];   // Hardware channel corresponding to TDC channel

  auto ref_beg = mReco[ibeg].ref[TDCSignal[itdc]];
  auto ref_end = mReco[iend].ref[TDCSignal[itdc]];

  O2_ZDC_DIGIRECO_FLT first_sample = mChData[ref_beg].data[0];
  O2_ZDC_DIGIRECO_FLT last_sample = mChData[ref_end].data[NTimeBinsPerBC - 1];

  // Constant extrapolation at the beginning and at the end of the array
  // Assign value of first sample
  for (int i = 0; i < tsnh; i++) {
    mReco[ibeg].inter[itdc][i] = first_sample;
  }
  // Assign value of last sample
  for (int i = ntot - tsnh; i < ntot; i++) {
    int isam = i % nsbun;
    mReco[iend].inter[itdc][isam] = last_sample;
  }
  // Interpolation between acquired points (n.b. loop from 0 to nint)
  for (int i = 0; i < nint; i++) {
    // Identification of the point to be assigned (need to add tsnh to identify the point)
    int ibun = ibeg + (i + tsnh) / nsbun;
    int isam = (i + tsnh) % nsbun;
    int im = i % TSN;
    if (im == 0) {
      // This is an acquired point
      int ip = (i / TSN) % NTimeBinsPerBC;
      int ib = ibeg + (i / TSN) / NTimeBinsPerBC;
      if (ib != ibun) {
        LOG(FATAL) << "ib=" << ib << " ibun=" << ibun;
        return;
      }
      mReco[ibun].inter[itdc][isam] = mChData[mReco[ibun].ref[TDCSignal[itdc]]].data[ip];
    } else {
      // Do the actual interpolation
      O2_ZDC_DIGIRECO_FLT y = 0;
      int ip = i / TSN;
      O2_ZDC_DIGIRECO_FLT sum = 0;
      for (int is = TSN - im, ii = ip - TSL + 1; is < NTS; is += TSN, ii++) {
        // Default is first point in the array
        O2_ZDC_DIGIRECO_FLT yy = first_sample;
        if (ii > 0) {
          if (ii < nsam) {
            int ip = ii % NTimeBinsPerBC;
            int ib = ibeg + ii / NTimeBinsPerBC;
            yy = mChData[mReco[ib].ref[TDCSignal[itdc]]].data[ip];
          } else {
            // Last acquired point
            yy = last_sample;
          }
        }
        sum += mTS[is];
        y += yy * mTS[is];
      }
      y = y / sum;
      mReco[ibun].inter[itdc][isam] = y;
    }
  }
  // Looking for a local maximum in a searching zone
  float amp = std::numeric_limits<float>::infinity(); // Amplitude to be stored
  int isam_amp = 0;                                   // Sample at maximum amplitude (relative to beginning of group)
  int ip_old = -1, ip_cur = -1, ib_cur = -1;          // Current and old points
  bool is_searchable = false;                         // Flag for point in the search zone for maximum amplitude
  bool was_searchable = false;                        // Flag for point in the search zone for maximum amplitude
  int ib[nsp] = {-1, -1, -1, -1, -1};
  int ip[nsp] = {-1, -1, -1, -1, -1};
  // N.B. Points at the extremes are constant therefore no local maximum
  // can occur in these two regions
  for (int i = 0; i < nint; i++) {
    int isam = i + tsnh;
    // Check if trigger is fired for this point
    // For the moment we don't take into account possible extensions of the search zone
    // ip_cur can span several bunches and is used just to identify transitions
    ip_cur = isam / TSN;
    // Speed up computation
    if (ip_cur != ip_old) {
      ip_old = ip_cur;
      for (int j = 0; j < 5; j++) {
        ib[j] = -1;
        ip[j] = -1;
      }
      // There are three possible triple conditions that involve current point (middle is current point)
      ip[2] = ip_cur % NTimeBinsPerBC;
      ib[2] = ibeg + ip_cur / NTimeBinsPerBC;
      ib_cur = ib[2];
      if (ip[2] > 0) {
        ip[1] = ip[2] - 1;
        ib[1] = ib[2];
      } else if (ip[2] == 0) {
        if (ib[2] > ibeg) {
          ib[1] = ib[2] - 1;
          ip[1] = MaxTimeBin;
        }
      }
      if (ip[1] > 0) {
        ip[0] = ip[1] - 1;
        ib[0] = ib[1];
      } else if (ip[1] == 0) {
        if (ib[1] > ibeg) {
          ib[0] = ib[1] - 1;
          ip[0] = MaxTimeBin;
        }
      }
      if (ip[2] < MaxTimeBin) {
        ip[3] = ip[2] + 1;
        ib[3] = ib[2];
      } else if (ip[2] == MaxTimeBin) {
        if (ib[2] < iend) {
          ib[3] = ib[2] + 1;
          ip[3] = 0;
        }
      }
      if (ip[3] < MaxTimeBin) {
        ip[4] = ip[3] + 1;
        ib[4] = ib[3];
      } else if (ip[3] == MaxTimeBin) {
        if (ib[3] < iend) {
          ib[4] = ib[3] + 1;
          ip[4] = 0;
        }
      }
      // meet the threshold condition
      was_searchable = is_searchable;
      // Search conditions with list of allowed patterns
      // No need to double check ib[?] and ip[?] because either we assign both or none
      uint16_t triggered = 0x0000;
      for (int j = 0; j < 5; j++) {
        if (ib[j] >= 0 && (mReco[ib[j]].fired[itdc] & mMask[ip[j]]) > 0) {
          triggered |= (0x1 << j);
        }
      }
      // Reject conditions:
      // 00000
      // 10001
      // One among 10000 and 00001
      // Accept conditions:
      constexpr uint16_t accept[14] = {
        //          0x01, // 00001 extend search zone before maximum
        0x02, // 00010
        0x04, // 00100
        0x08, // 01000
        0x10, // 10000 extend after
        0x03, // 00011
        0x06, // 00110
        0x0c, // 01100
        0x18, // 11000
        0x07, // 00111
        0x0e, // 01110
        0x1c, // 11100
        0x0f, // 01111
        0x1e, // 11110
        0x1f  // 11111
      };
      // All other are not correct (->reject)
      is_searchable = 0;
      if (triggered != 0) {
        for (int j = 0; j < 14; j++) {
          if (triggered == accept[j]) {
            is_searchable = 1;
            break;
          }
        }
      }
    }
    // We do not restrict search zone around expected main-main collision
    // because we would like to be able to identify pile-up from collisions
    // with satellites (buggy)

    // If we exit from searching zone
    if (was_searchable && !is_searchable) {
      if (amp <= ADCMax) {
        // Store identified peak
        int ibun = ibeg + isam_amp / nsbun;
        int tdc = isam_amp % nsbun;
        // Look for offset
        auto orbit = mBCData[ibun].ir.orbit;
        std::map<uint32_t, int>::iterator it = mOrbit.find(orbit);
        if (it != mOrbit.end()) {
          // Subtract pedestal
          auto& orbitdata = mOrbitData[it->second];
          amp = orbitdata.asFloat(ich) - amp;
        } else {
          LOG(ERROR) << "Missing pedestal";
          amp = std::numeric_limits<float>::infinity();
        }
        assignTDC(ibun, ibeg, iend, itdc, tdc, amp);
      }
      amp = std::numeric_limits<float>::infinity();
      isam_amp = 0;
      was_searchable = 0;
    }
    if (is_searchable) {
      int mysam = isam % nsbun;
      if (mReco[ib_cur].inter[itdc][mysam] < amp) {
        amp = mReco[ib_cur].inter[itdc][mysam];
        isam_amp = isam;
      }
    }
  }
  // Trigger flag still present at the of the scan
  if (is_searchable) {
    // Add last identified peak
    if (amp <= ADCMax) {
      // Store identified peak
      int ibun = ibeg + isam_amp / nsbun;
      int tdc = isam_amp % nsbun;
      // Look for offset
      auto orbit = mBCData[ibun].ir.orbit;
      std::map<uint32_t, int>::iterator it = mOrbit.find(orbit);
      if (it != mOrbit.end()) {
        // Subtract pedestal
        auto& orbitdata = mOrbitData[it->second];
        amp = orbitdata.asFloat(ich) - amp;
      } else {
        LOG(ERROR) << "Missing pedestal";
        amp = std::numeric_limits<float>::infinity();
      }
      assignTDC(ibun, ibeg, iend, itdc, tdc, amp);
    }
  }
}

void DigiReco::assignTDC(int ibun, int ibeg, int iend, int itdc, int tdc, float amp)
{
  constexpr int nsbun = TSN * NTimeBinsPerBC; // Total number of interpolated points per bunch crossing
  constexpr int tdc_max = nsbun / 2;
  constexpr int tdc_min = -tdc_max;

  // Apply tdc shift correction
  int tdc_cor = tdc - tdc_shift[itdc];
  // Correct bunch assignment
  if (tdc_cor < tdc_min && ibun >= ibeg) {
    // Assign to preceding bunch
    ibun = ibun - 1;
    tdc_cor = tdc_cor + nsbun;
  } else if (tdc_cor >= tdc_max && ibun < iend) {
    // Assign to following bunch
    ibun = ibun + 1;
    tdc_cor = tdc_cor - nsbun;
  }
  if (tdc_cor < kMinShort) {
    LOG(ERROR) << "TDC " << itdc << " " << tdc_cor << " is out of range";
    tdc_cor = kMinShort;
  }
  if (tdc_cor > kMaxShort) {
    LOG(ERROR) << "TDC " << itdc << " " << tdc_cor << " is out of range";
    tdc_cor = kMaxShort;
  }
  // Assign to correct bunch
  int& ihit = mReco[ibun].ntdc[itdc];
  if (ihit < MaxTDCValues) {
    mReco[ibun].tdcVal[itdc][ihit] = tdc_cor;
    mReco[ibun].tdcAmp[itdc][ihit] = std::nearbyint(amp / FTDCAmp);
    ihit++;
#ifdef O2_ZDC_DEBUG
    LOG(INFO) << mReco[ibun].ir.orbit << "." << mReco[ibun].ir.bc << " "
              << "ibun=" << ibun << " itdc=" << itdc << " tdc=" << tdc << " tdc_cor=" << tdc_cor * FTDCVal << " amp=" << amp * FTDCAmp;
#endif
  } else {
    LOG(ERROR) << mReco[ibun].ir.orbit << "." << mReco[ibun].ir.bc << " "
               << "ibun=" << ibun << " itdc=" << itdc << " tdc=" << tdc << " tdc_cor=" << tdc_cor * FTDCVal << " amp=" << amp * FTDCAmp << " OVERFLOW";
  }
}

} // namespace zdc
} // namespace o2
