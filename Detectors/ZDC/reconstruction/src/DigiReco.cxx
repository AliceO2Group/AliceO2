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

void DigiReco::init()
{
  LOG(info) << "Initialization of ZDC reconstruction";
  // Load configuration parameters
  auto& sopt = ZDCSimParam::Instance();
  mIsContinuous = sopt.continuous;
  mNBCAHead = mIsContinuous ? sopt.nBCAheadCont : sopt.nBCAheadTrig;

  if (!mModuleConfig) {
    LOG(fatal) << "Missing ModuleConfig configuration object";
    return;
  }

  mTriggerMask = mModuleConfig->getTriggerMask();

  prepareInterpolation();

  if (mTreeDbg) {
    // Open debug file
    LOG(info) << "ZDC DigiReco: opening debug output";
    mDbg = std::make_unique<TFile>("ZDCRecoDbg.root", "recreate");
    mTDbg = std::make_unique<TTree>("zdcr", "ZDCReco");
    mTDbg->Branch("zdcr", "RecEventAux", &mRec);
  }

  // Update reconstruction parameters
  // auto& ropt=RecoParamZDC::Instance();
  o2::zdc::RecoParamZDC& ropt = const_cast<o2::zdc::RecoParamZDC&>(RecoParamZDC::Instance());
  ropt.print();
  mRopt = (o2::zdc::RecoParamZDC*)&ropt;

  // Fill maps to decode the pattern of channels with hit
  for (int itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    // If the reconstruction parameters were not manually set
    if (ropt.tmod[itdc] < 0 || ropt.tch[itdc] < 0) {
      int isig = TDCSignal[itdc];
      for (int im = 0; im < NModules; im++) {
        for (uint32_t ic = 0; ic < NChPerModule; ic++) {
          if (mModuleConfig->modules[im].channelID[ic] == isig && mModuleConfig->modules[im].readChannel[ic]) {
            // ropt.updateFromString(TString::Format("RecoParamZDC.tmod[%d]=%d;",itdc,im));
            // ropt.updateFromString(TString::Format("RecoParamZDC.tch[%d]=%d;",itdc,ic));
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
    if (mVerbosity > DbgZero) {
      LOG(info) << "TDC " << itdc << "(" << ChannelNames[TDCSignal[itdc]] << ")"
                << " mod " << ropt.tmod[itdc] << " ch " << ropt.tch[itdc];
    }
  }

  // Signal processing
  // N.B. Function call overrides other settings for low pass filtering
  if (mLowPassFilterSet == false) {
    if (ropt.low_pass_filter < 0) {
      if (!mRecoConfigZDC) {
        LOG(fatal) << "Configuration of low pass filtering: missing configuration object and no manual override";
      } else {
        ropt.low_pass_filter = mRecoConfigZDC->low_pass_filter;
      }
    }
    mLowPassFilter = ropt.low_pass_filter > 0 ? true : false;
  }
  LOG(info) << "Low pass filtering is " << (mLowPassFilter ? "enabled" : "disabled");

  // Full interpolation of waveform (N.B. function call overrides other settings)
  if (mFullInterpolationSet == false) {
    if (ropt.full_interpolation < 0) {
      if (!mRecoConfigZDC) {
        LOG(fatal) << "Configuration of interpolation: missing configuration object and no manual override";
      } else {
        ropt.full_interpolation = mRecoConfigZDC->full_interpolation;
      }
    }
    mFullInterpolation = ropt.full_interpolation > 0 ? true : false;
  }
  if (ropt.full_interpolation_min_length >= 2) {
    // Minimum is 2 consecutive bunch crossings
    mFullInterpolationMinLength = ropt.full_interpolation_min_length;
  }
  if (mVerbosity > DbgZero) {
    LOG(info) << "Full waveform interpolation is " << (mFullInterpolation ? "enabled" : "disabled") << " min length is " << mFullInterpolationMinLength;
  }

  if (ropt.triggerCondition == 0) {
    if (!mRecoConfigZDC) {
      LOG(fatal) << "Trigger condition: missing configuration object and no manual override";
    } else {
      mTriggerCondition = mRecoConfigZDC->triggerCondition;
    }
  } else {
    mTriggerCondition = ropt.triggerCondition;
  }

  if (mTriggerCondition == 0x1) {
    LOG(info) << "SINGLE trigger condition enforced in reconstruction";
  } else if (mTriggerCondition == 0x3) {
    LOG(info) << "DOUBLE trigger condition enforced in reconstruction";
  } else if (mTriggerCondition == 0x7) {
    LOG(info) << "TRIPLE trigger condition enforced in reconstruction";
  } else {
    LOG(fatal) << "Unsupported trigger condition in ZDC reconstruction: " << mTriggerCondition;
  }

  if (mCorrSignalSet == false) {
    if (ropt.corr_signal < 0) {
      if (!mRecoConfigZDC) {
        LOG(fatal) << "Configuration of TDC signal correction: missing configuration object and no manual override";
      } else {
        ropt.corr_signal = mRecoConfigZDC->corr_signal;
      }
    }
    mCorrSignal = ropt.corr_signal > 0 ? true : false;
  }
  if (mVerbosity > DbgZero) {
    LOG(info) << "TDC signal correction is " << (mCorrSignal ? "enabled" : "disabled");
  }

  if (mCorrBackgroundSet == false) {
    if (ropt.corr_background < 0) {
      if (!mRecoConfigZDC) {
        LOG(fatal) << "Configuration of TDC pile-up correction: missing configuration object and no manual override";
      } else {
        ropt.corr_background = mRecoConfigZDC->corr_background;
      }
    }
    mCorrBackground = ropt.corr_background > 0 ? true : false;
  }
  if (mVerbosity > DbgZero) {
    LOG(info) << "TDC pile-up correction is " << (mCorrBackground ? "enabled" : "disabled");
  }

  // Trigger configuration
  for (int itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    // If the reconstruction parameters were not manually set
    if (ropt.tth[itdc] <= 0) {
      if (!mRecoConfigZDC) {
        LOG(fatal) << "Trigger threshold for TDC " << itdc << " missing configuration object and no manual override";
      } else {
        ropt.tth[itdc] = mRecoConfigZDC->tth[itdc];
      }
    }
    if (ropt.tsh[itdc] <= 0) {
      if (!mRecoConfigZDC) {
        LOG(fatal) << "Trigger shift for TDC " << itdc << " missing configuration object and no manual override";
      } else {
        ropt.tsh[itdc] = mRecoConfigZDC->tsh[itdc];
      }
    }
    if (mVerbosity > DbgZero) {
      LOG(info) << itdc << " " << ChannelNames[TDCSignal[itdc]] << " tth= " << mRopt->tth[itdc] << " tsh= " << mRopt->tsh[itdc];
    }
  }

  // Extended search configuration
  if (ropt.setExtendedSearch == false) {
    if (mRecoConfigZDC == nullptr) {
      LOG(fatal) << "Extended search configuration: missing configuration object and no manual override";
    } else {
      ropt.doExtendedSearch = mRecoConfigZDC->extendedSearch;
    }
  }
  LOG(info) << "Extended search is " << (mRopt->doExtendedSearch ? "ON" : "OFF");

  // Store hits with in-event pile-up (EvE)
  if (ropt.setStoreEvPileup == false) {
    if (mRecoConfigZDC == nullptr) {
      LOG(fatal) << "Storage of hits with in-event pile-up: missing configuration object and no manual override";
    } else {
      ropt.doStoreEvPileup = mRecoConfigZDC->storeEvPileup;
    }
  }
  LOG(info) << "Storage of EvE hits is " << (mRopt->doStoreEvPileup ? "ON" : "OFF");

  // TDC calibration
  // Recentering
  for (int itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    float fval = ropt.tdc_shift[itdc];
    // If the reconstruction parameters were not manually set
    if (fval < 0) {
      // Check if calibration object is present
      if (!mTDCParam) {
        LOG(fatal) << "TDC " << itdc << " missing configuration object and no manual override";
      } else {
        // Encode TDC shift
        fval = mTDCParam->getShift(itdc) / o2::zdc::FTDCVal;
      }
    }
    auto val = std::nearbyint(fval);
    if (val < kMinShort) {
      LOG(fatal) << "Shift for TDC " << itdc << " " << val << " is out of range";
    }
    if (val > kMaxShort) {
      LOG(fatal) << "Shift for TDC " << itdc << " " << val << " is out of range";
    }
    tdc_shift[itdc] = val;
    if (mVerbosity > DbgZero) {
      LOG(info) << itdc << " " << ChannelNames[TDCSignal[itdc]] << " shift= " << tdc_shift[itdc] << " i.s. = " << val * o2::zdc::FTDCVal << " ns";
    }
  }

  // Amplitude calibration
  for (int itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    float fval = ropt.tdc_calib[itdc];
    // If the reconstruction parameters were not manually set
    if (fval < 0) {
      // Check if calibration object is present
      if (!mTDCParam) {
        LOG(fatal) << "TDC " << itdc << " missing configuration object and no manual override";
      } else {
        fval = mTDCParam->getFactor(itdc);
      }
    }
    if (fval <= 0) {
      LOG(fatal) << "Correction factor for TDC amplitude " << itdc << " " << fval << " is out of range";
    }
    tdc_calib[itdc] = fval;
    if (mVerbosity > DbgZero) {
      LOG(info) << itdc << " " << ChannelNames[TDCSignal[itdc]] << " factor= " << tdc_calib[itdc];
    }
  }

  // TDC search zone
  for (int itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    // If the reconstruction parameters were not manually set
    if (ropt.tdc_search[itdc] <= 0) {
      if (!mRecoConfigZDC) {
        LOG(fatal) << "Search zone for TDC " << itdc << " missing configuration object and no manual override";
      } else {
        ropt.tdc_search[itdc] = mRecoConfigZDC->tdc_search[itdc];
      }
    }
    if (mVerbosity > DbgZero) {
      LOG(info) << itdc << " " << ChannelNames[TDCSignal[itdc]] << " search= " << ropt.tdc_search[itdc] << " i.s. = " << ropt.tdc_search[itdc] * o2::zdc::FTDCVal << " ns";
    }
  }

  // Energy calibration
  for (int il = 0; il < ChEnergyCalib.size(); il++) {
    if (ropt.energy_calib[ChEnergyCalib[il]] > 0) {
      LOG(info) << "Energy Calibration from command line " << ChannelNames[ChEnergyCalib[il]] << " = " << ropt.energy_calib[ChEnergyCalib[il]];
    } else if (mEnergyParam && mEnergyParam->energy_calib[ChEnergyCalib[il]] > 0) {
      ropt.energy_calib[ChEnergyCalib[il]] = mEnergyParam->energy_calib[ChEnergyCalib[il]];
      LOG(info) << "Energy Calibration from CCDB " << ChannelNames[ChEnergyCalib[il]] << " = " << ropt.energy_calib[ChEnergyCalib[il]];
    } else {
      if (ChEnergyCalib[il] == CaloCommonPM[ChEnergyCalib[il]]) {
        // Is a common PM or a ZEM
        ropt.energy_calib[ChEnergyCalib[il]] = 1;
        LOG(warning) << "Default Energy Calibration  " << ChannelNames[ChEnergyCalib[il]] << " = " << ropt.energy_calib[ChEnergyCalib[il]];
      } else {
        // Is one of the analog sums -> same calibration as common PM
        // N.B. the calibration for common has already been set in the loop
        ropt.energy_calib[ChEnergyCalib[il]] = ropt.energy_calib[CaloCommonPM[il]];
        LOG(info) << "SUM Energy Calibration  " << ChannelNames[ChEnergyCalib[il]] << " = " << ropt.energy_calib[ChEnergyCalib[il]];
      }
    }
  }

  // Tower calibration (intercalibration of towers)
  for (int il = 0; il < ChTowerCalib.size(); il++) {
    if (ropt.tower_calib[ChTowerCalib[il]] > 0) {
      LOG(info) << "Tower Calibration from command line " << ChannelNames[ChTowerCalib[il]] << " = " << ropt.tower_calib[ChTowerCalib[il]];
    } else if (mTowerParam && mTowerParam->tower_calib[ChTowerCalib[il]] > 0) {
      ropt.tower_calib[ChTowerCalib[il]] = mTowerParam->tower_calib[ChTowerCalib[il]];
      LOG(info) << "Tower Calibration from CCDB " << ChannelNames[ChTowerCalib[il]] << " = " << ropt.tower_calib[ChTowerCalib[il]];
    } else {
      ropt.tower_calib[ChTowerCalib[il]] = 1;
      LOG(warning) << "Default Tower Calibration " << ChannelNames[ChTowerCalib[il]] << " = " << ropt.tower_calib[ChTowerCalib[il]];
    }
  }

  // Tower energy calibration
  // For the common PM and sum is given by the calibration parameters
  // for the towers is the product of energy calibration of common PM and tower calibration
  // It is possible to override from command line
  for (int il = 0; il < ChTowerCalib.size(); il++) {
    if (ropt.energy_calib[ChTowerCalib[il]] > 0) {
      LOG(info) << "Tower Energy Calibration set to " << ChannelNames[ChTowerCalib[il]] << " = " << ropt.energy_calib[ChTowerCalib[il]];
    } else {
      ropt.energy_calib[ChTowerCalib[il]] = ropt.tower_calib[ChTowerCalib[il]] * ropt.energy_calib[CaloCommonPM[ChTowerCalib[il]]];
      if (mVerbosity > DbgZero) {
        LOG(info) << "Tower Energy Calibration " << ChannelNames[ChTowerCalib[il]] << " = " << ropt.energy_calib[ChTowerCalib[il]];
      }
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
            // Fill mask to identify all channels
            mChMask[ich] = (0x1 << (4 * im + ic));
            goto next_ich;
          }
        }
      }
    }
  next_ich:;
    if (mVerbosity > DbgZero) {
      LOG(info) << "ADC " << ich << "(" << ChannelNames[ich] << ") mod " << ropt.amod[ich] << " ch " << ropt.ach[ich];
    }
  }

  // Integration ranges
  for (int ich = 0; ich < NChannels; ich++) {
    // If the reconstruction parameters were not manually set
    if (ropt.beg_int[ich] == DummyIntRange || ropt.end_int[ich] == DummyIntRange) {
      if (!mRecoConfigZDC) {
        LOG(fatal) << "Integration for signal " << ich << " missing configuration object and no manual override";
      } else {
        ropt.beg_int[ich] = mRecoConfigZDC->beg_int[ich];
        ropt.end_int[ich] = mRecoConfigZDC->end_int[ich];
      }
    }
    if (ropt.beg_ped_int[ich] == DummyIntRange || ropt.end_ped_int[ich] == DummyIntRange) {
      if (!mRecoConfigZDC) {
        LOG(fatal) << "Integration for pedestal " << ich << " missing configuration object and no manual override";
      } else {
        ropt.beg_ped_int[ich] = mRecoConfigZDC->beg_ped_int[ich];
        ropt.end_ped_int[ich] = mRecoConfigZDC->end_ped_int[ich];
      }
    }
    // Thresholds for pedestal
    // If the reconstruction parameters were not manually set
    if (ropt.beg_int[ich] == ADCRange) {
      if (!mRecoConfigZDC) {
        LOG(fatal) << "Pedestal threshold high for signal " << ich << " missing configuration object and no manual override";
      } else {
        ropt.ped_thr_hi[ich] = mRecoConfigZDC->ped_thr_hi[ich];
      }
    }
    if (ropt.beg_int[ich] == ADCRange) {
      if (!mRecoConfigZDC) {
        LOG(fatal) << "Pedestal threshold low for signal " << ich << " missing configuration object and no manual override";
      } else {
        ropt.ped_thr_lo[ich] = mRecoConfigZDC->ped_thr_lo[ich];
      }
    }
    if (mVerbosity > DbgZero) {
      LOG(info) << ChannelNames[ich] << " integration: signal=[" << ropt.beg_int[ich] << ":" << ropt.end_int[ich] << "] pedestal=[" << ropt.beg_ped_int[ich] << ":" << ropt.end_ped_int[ich]
                << "] thresholds (" << ropt.ped_thr_hi[ich] << ", " << ropt.ped_thr_lo[ich] << ")";
    }
  }
  if (mVerbosity > DbgZero && mTDCCorr != nullptr) {
    mTDCCorr->print();
  }
  // Information about reconstruction configuration
  if (mLowPassFilter == false) {
    LOG(warning) << "Low pass filtering of signal is disabled";
  }
  if (mCorrSignal == false) {
    LOG(warning) << "TDC signal correction is disabled";
  }
  if (mCorrBackground == false) {
    LOG(warning) << "TDC pile-up correction is disabled";
  }
  if (mPedParam == nullptr) {
    LOG(warning) << "Missing average pedestals";
  } else {
    if (mVerbosity > DbgZero) {
      mPedParam->print();
    } else {
      mPedParam->print(false);
    }
  }
} // init

void DigiReco::eor()
{
  if (mTreeDbg) {
    LOG(info) << "o2::zdc::DigiReco: closing debug output";
    mTDbg->Write();
    mTDbg.reset();
    mDbg->Close();
    mDbg.reset();
  }

  ZDCTDCDataErr::print();

  if (mNLonely > 0) {
    LOG(warn) << "Detected " << mNLonely << " lonely bunches";
    for (int ib = 0; ib < o2::constants::lhc::LHCMaxBunches; ib++) {
      if (mLonely[ib]) {
        LOGF(warn, "lonely bunch %4d #times=%u #trig=%u", ib, mLonely[ib], mLonelyTrig[ib]);
      }
    }
  }
  for (int ich = 0; ich < NChannels; ich++) {
    if (mMissingPed[ich] > 0) {
      LOGF(error, "Missing pedestal for ch %2d %s: %u", ich, ChannelNames[ich], mMissingPed[ich]);
    }
  }
}

void DigiReco::prepareInterpolation()
{
  // Prepare tapered sinc function
  // Lost reference of first interpolating function
  // Found problems in implementation: the interpolated function is not derivable in sampled points
  // tsc/TSN =3.75 (~ 4) and TSL*TSN*sqrt(2)/tsc >> 1 (n. of sigma)
  // const O2_ZDC_DIGIRECO_FLT tsc = 750;
  // Now using Kaiser function
  O2_ZDC_DIGIRECO_FLT beta = TMath::Pi() * mAlpha;
  O2_ZDC_DIGIRECO_FLT norm = 1. / TMath::BesselI0(beta);
  constexpr int n = TSL * TSN;
  for (int tsi = 0; tsi <= n; tsi++) {
    O2_ZDC_DIGIRECO_FLT arg1 = TMath::Pi() * O2_ZDC_DIGIRECO_FLT(tsi) / O2_ZDC_DIGIRECO_FLT(TSN);
    O2_ZDC_DIGIRECO_FLT fs = 1;
    if (arg1 != 0) {
      fs = TMath::Sin(arg1) / arg1;
    }
    // First tapering window
    // O2_ZDC_DIGIRECO_FLT arg2 = O2_ZDC_DIGIRECO_FLT(tsi) / tsc;
    // O2_ZDC_DIGIRECO_FLT fg = TMath::Exp(-arg2 * arg2);
    // Kaiser window
    O2_ZDC_DIGIRECO_FLT arg2 = O2_ZDC_DIGIRECO_FLT(tsi) / O2_ZDC_DIGIRECO_FLT(n);
    O2_ZDC_DIGIRECO_FLT fg = norm * TMath::BesselI0(beta * TMath::Sqrt(1. - arg2 * arg2));
    mTS[n + tsi] = fs * fg;
    mTS[n - tsi] = mTS[n + tsi]; // Function is even
  }
  LOG(info) << "Interpolation numeric precision is " << sizeof(O2_ZDC_DIGIRECO_FLT);
  LOG(info) << "Interpolation alpha = " << mAlpha;
}

int DigiReco::process(const gsl::span<const o2::zdc::OrbitData>& orbitdata, const gsl::span<const o2::zdc::BCData>& bcdata, const gsl::span<const o2::zdc::ChannelData>& chdata)
{
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  LOG(info) << "________________________________________________________________________________";
  LOG(info) << __func__;
#endif
  // We assume that vectors contain data from a full time frame
  mOrbitData = orbitdata;
  mBCData = bcdata;
  mChData = chdata;
  mInError = false;

  // Initialization of lookup structure for pedestals
  mOrbit.clear();
  int norb = mOrbitData.size();
  if (mVerbosity >= DbgFull) {
    LOG(info) << "Dump of pedestal data lookup table";
  }
  // TODO: send scalers to aggregator
  uint32_t scaler[NChannels] = {0};
  for (int iorb = 0; iorb < norb; iorb++) {
    mOrbit[mOrbitData[iorb].ir.orbit] = iorb;
    if (mVerbosity >= DbgFull) {
      LOG(info) << "mOrbitData[" << mOrbitData[iorb].ir.orbit << "] = " << iorb;
    }
    // TODO: add only if orbit is good. Here we check only the individual scaler values
    for (int ich = 0; ich < NChannels; ich++) {
      if (mOrbitData[iorb].scaler[ich] <= o2::constants::lhc::LHCMaxBunches) {
        scaler[ich] += mOrbitData[iorb].scaler[ich];
      }
    }
  }
  if (mVerbosity > DbgMinimal) {
    for (int ich = 0; ich < NChannels; ich++) {
      LOG(info) << ChannelNames[ich] << " cnt: " << scaler[ich];
    }
  }

  mNBC = mBCData.size();
  mReco.clear();
  mReco.resize(mNBC);
  // Initialization of reco structure
  for (int ibc = 0; ibc < mNBC; ibc++) {
    auto& bcr = mReco[ibc];
#ifdef O2_ZDC_TDC_C_ARRAY
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
      for (int i = 0; i < MaxTDCValues; i++) {
        bcr.tdcVal[itdc][i] = kMinShort;
        bcr.tdcAmp[itdc][i] = kMinShort;
      }
    }
#endif
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
  //   for(int isig=0; isig<NChannels; isig++){
  //     mReco.pattern[isig]=0;
  //     for(int itb=0; itb<NTimeBinsPerBC; itb++){
  //       mReco.fired[isig][itb]=0;
  //     }
  //     for(int isb=0; isb<mNSB; isb++){
  //       mReco.inter[isig][isb]=0;
  //     }
  //   }

  // Assign interaction record and event information
  for (int ibc = 0; ibc < mNBC; ibc++) {
    mReco[ibc].ir = mBCData[ibc].ir;
    mReco[ibc].channels = mBCData[ibc].channels;
    mReco[ibc].triggers = mBCData[ibc].triggers;
    for (int isig = 0; isig < NChannels; isig++) {
      auto ref = mReco[ibc].ref[isig];
      if (ref == ZDCRefInitVal) {
        for (int is = 0; is < NTimeBinsPerBC; is++) {
          mReco[ibc].data[isig][is] = Int16MaxVal;
        }
      } else {
        for (int is = 0; is < NTimeBinsPerBC; is++) {
          mReco[ibc].data[isig][is] = mChData[ref].data[is];
        }
      }
    }
  }

  // Low pass filtering
  if (mLowPassFilter) {
    // N.B. At the moment low pass filtering is performed only on TDC
    // signals and not on the rest of the signals
    lowPassFilter();
  } else {
    // Copy samples
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
      auto isig = TDCSignal[itdc];
      for (int ibc = 0; ibc < mNBC; ibc++) {
        auto ref_c = mReco[ibc].ref[isig];
        if (ref_c != ZDCRefInitVal) {
          for (int is = 0; is < NTimeBinsPerBC; is++) {
            mReco[ibc].data[isig][is] = mChData[ref_c].data[is];
          }
        }
      }
    }
  }

  if (mFullInterpolation) {
    // Copy remaining channels
    for (int isig = 0; isig < NChannels; isig++) {
      int isig_tdc = TDCSignal[SignalTDC[isig]];
      if (isig == isig_tdc) {
        // Already copied
        continue;
      }
      for (int ibc = 0; ibc < mNBC; ibc++) {
        auto ref_c = mReco[ibc].ref[isig];
        if (ref_c != ZDCRefInitVal) {
          for (int is = 0; is < NTimeBinsPerBC; is++) {
            mReco[ibc].data[isig][is] = mChData[ref_c].data[is];
          }
        }
      }
    }
  }

  // Find consecutive bunch crossings by taking into account just the presence
  // of bunch crossing data and then perform signal interpolation in the identified ranges.
  // With this definition of "consecutive" bunch crossings gaps in the sample data
  // may be present, therefore in the reconstruction method we take into account for signals
  // that do not span the entire range
  int seq_beg = 0;
  int seq_end = 0;
  if (mVerbosity > DbgMinimal) {
    LOG(info) << "Processing ZDC reconstruction for " << mNBC << " bunch crossings";
  }

  // TDC reconstruction
  for (int ibc = 0; ibc < mNBC; ibc++) {
    auto& ir = mBCData[seq_end].ir;
    auto bcd = mBCData[ibc].ir.differenceInBC(ir);
    if (bcd < 0) {
      LOG(error) << "Bunch order error in TDC reconstruction";
      for (int ibcdump = 0; ibcdump < mNBC; ibcdump++) {
        LOG(error) << "mBCData[" << ibcdump << "] @ " << mBCData[ibcdump].ir.orbit << "." << mBCData[ibcdump].ir.bc;
      }
      LOG(error) << "Orbit number is not increasing " << mBCData[seq_end].ir.orbit << "." << mBCData[seq_end].ir.bc << " followed by " << mBCData[ibc].ir.orbit << "." << mBCData[ibc].ir.bc;
      return __LINE__;
    } else if (bcd > 1) {
      // Detected a gap
      int rval = reconstructTDC(seq_beg, seq_end);
      if (rval) {
        return rval;
      }

      seq_beg = ibc;
      seq_end = ibc;
    } else if (ibc == (mNBC - 1)) {
      // Last bunch
      seq_end = ibc;
      int rval = reconstructTDC(seq_beg, seq_end);
      if (rval) {
        return rval;
      }

      seq_beg = mNBC;
      seq_end = mNBC;
    } else {
      // Look for another bunch
      seq_end = ibc;
    }
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
    // Here in order to avoid mixing information
    mBCData[ibc].print(mTriggerMask);
#endif
  }

  // Apply pile-up correction for TDCs to get corrected TDC amplitudes and values
  correctTDCPile();

  // ADC reconstruction
  seq_beg = 0;
  seq_end = 0;
  for (int ibc = 0; ibc < mNBC; ibc++) {
    auto& ir = mBCData[seq_end].ir;
    auto bcd = mBCData[ibc].ir.differenceInBC(ir);
    if (bcd < 0) {
      LOG(error) << "Bunch order error in ADC reconstruction";
      for (int ibcdump = 0; ibcdump < mNBC; ibcdump++) {
        LOG(error) << "mBCData[" << ibcdump << "] @ " << mBCData[ibcdump].ir.orbit << "." << mBCData[ibcdump].ir.bc;
      }
      LOG(error) << "Orbit number is not increasing " << mBCData[seq_end].ir.orbit << "." << mBCData[seq_end].ir.bc << " followed by " << mBCData[ibc].ir.orbit << "." << mBCData[ibc].ir.bc;
      return __LINE__;
    } else if (bcd > 1) {
      // Detected a gap
      int rval = reconstruct(seq_beg, seq_end);
      if (rval != 0) {
        return rval;
      }
      seq_beg = ibc;
      seq_end = ibc;
    } else if (ibc == (mNBC - 1)) {
      // Last bunch
      seq_end = ibc;
      int rval = reconstruct(seq_beg, seq_end);
      if (rval != 0) {
        return rval;
      }
      seq_beg = mNBC;
      seq_end = mNBC;
    } else {
      // Look for another bunch
      seq_end = ibc;
    }
  }
  return 0;
} // process

void DigiReco::lowPassFilter()
{
  // First attempt to low pass filtering uses the average of three consecutive samples
  // ringing noise has T~6 ns w.r.t. a sampling period of ~ 2 ns
  // one should get smoothing of the noise
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  LOG(info) << "________________________________________________________________________________";
  LOG(info) << __func__;
#endif
  constexpr int MaxTimeBin = NTimeBinsPerBC - 1;
  for (int itdc = 0; itdc < NTDCChannels; itdc++) {
    auto isig = TDCSignal[itdc];
    for (int ibc = 0; ibc < mNBC; ibc++) {
      // Indexes of current, previous and next recorded bunch crossings
      auto ref_c = mReco[ibc].ref[isig];
      uint32_t ref_p = ZDCRefInitVal;
      uint32_t ref_n = ZDCRefInitVal;
      int64_t bcd_p = ZDCRefInitVal;
      int64_t bcd_n = ZDCRefInitVal;
      if (ibc > 0) { // Is not first bunch in list
        ref_p = mReco[ibc - 1].ref[isig];
        bcd_p = mReco[ibc].ir.differenceInBC(mReco[ibc - 1].ir); // b.c. number of (ibc) -  b.c. number (ibc-1)
      }
      if (ibc < (mNBC - 1)) { // Is not last bunch in list
        ref_n = mReco[ibc + 1].ref[isig];
        bcd_n = mReco[ibc + 1].ir.differenceInBC(mReco[ibc].ir); // b.c. number of (ibc+1) -  b.c. number (ibc)
      }
      if (ref_c != ZDCRefInitVal) { // Should always be true
        for (int is = 0; is < NTimeBinsPerBC; is++) {
          int32_t sum = mChData[ref_c].data[is];
          if (is == 0) {
            sum += mChData[ref_c].data[1];
            if (ref_p != ZDCRefInitVal && bcd_p == 1) {
              // Add last sample of previous bunch crossing
              sum += mChData[ref_p].data[MaxTimeBin];
            } else {
              // As a backup we count twice the first sample
              sum += mChData[ref_c].data[0];
            }
          } else if (is == MaxTimeBin) {
            sum += mChData[ref_c].data[MaxTimeBin - 1];
            if (ref_n != ZDCRefInitVal && bcd_n == 1) {
              // Add first sample of next bunch crossing
              sum += mChData[ref_n].data[0];
            } else {
              // As a backup we count twice the last sample
              sum += mChData[ref_c].data[MaxTimeBin];
            }
          } else {
            // Not on the bunch boundary
            sum += mChData[ref_c].data[is - 1];
            sum += mChData[ref_c].data[is + 1];
          }
          // Make the average taking into account rounding and sign
          bool isNegative = sum < 0;
          if (isNegative) {
            sum = -sum;
          }
          auto mod = sum % 3;
          sum = sum / 3;
          if (mod == 2) {
            sum++;
          }
          if (isNegative) {
            sum = -sum;
          }
          // Store filtered values
          mReco[ibc].data[isig][is] = sum;
        }
      }
    }
  }
}

int DigiReco::reconstructTDC(int ibeg, int iend)
{
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  LOG(info) << "________________________________________________________________________________";
  LOG(info) << __func__ << "(" << ibeg << ", " << iend << ") length=" << iend - ibeg + 1;
  for (int itdc = 0; itdc < NTDCChannels; itdc++) {
    mAssignedTDC[itdc] = 0;
  }
#endif
  // Apply differential discrimination
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
          int rval = 0;
          if (mRopt->doExtendedSearch) {
            rval = processTriggerExtended(itdc, istart, istop);
          } else {
            rval = processTrigger(itdc, istart, istop);
          }
          if (rval) {
            return rval;
          }
        }
        istart = -1;
        istop = -1;
      }
    }
    // Check if there are consecutive bunch crossings at the end of group
    if (istart >= 0 && (istop - istart) > 0) {
      int rval = 0;
      if (mRopt->doExtendedSearch) {
        rval = processTriggerExtended(itdc, istart, istop);
      } else {
        rval = processTrigger(itdc, istart, istop);
      }
      if (rval) {
        return rval;
      }
    }
  }
  // processTrigger(..) calls interpolate(..) that assigns all TDCs
  // The following TDC processing stage findSignals(..) assumes that time shift
  // due to pile-up has been corrected because main-main is assumed to be
  // in position 0
  // In case we do full interpolation, we process also the channels that are not
  // considered in TDC list
  if (mFullInterpolation) {
    for (int isig = 0; isig < NChannels; isig++) {
      int isig_tdc = TDCSignal[SignalTDC[isig]];
      if (isig == isig_tdc) {
        // Already computed
        continue;
      }
      // Check if channel has valid data for consecutive bunches in current bunch range
      // N.B. there are events recorded from ibeg-iend but we are not sure if it is the
      // case for every channel
      int istart = -1, istop = -1;
      // Loop allows for gaps in the data sequence for each TDC channel
      for (int ibun = ibeg; ibun <= iend; ibun++) {
        if (mBCData[ibun].channels & mChMask[isig]) { // Channel has data for this event
          if (istart < 0) {
            istart = ibun;
          }
          istop = ibun;
        } else { // No data from channel
          // A gap is detected
          if (istart >= 0 && (istop - istart + 1) >= mFullInterpolationMinLength) {
            // Need data for at least mFullInterpolationMinLength (two) consecutive bunch crossings
            int rval = fullInterpolation(isig, istart, istop);
            if (rval) {
              return rval;
            }
          }
          istart = -1;
          istop = -1;
        }
      }
      // Check if there are mFullInterpolationMinLength consecutive bunch crossings at the end of group
      if (istart >= 0 && (istop - istart + 1) >= mFullInterpolationMinLength) {
        int rval = fullInterpolation(isig, istart, istop);
        if (rval) {
          return rval;
        }
      }
    }
  }
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  printf("Assiged TDCs:");
  bool hasMult = false;
  for (int itdc = 0; itdc < NTDCChannels; itdc++) {
    if (mAssignedTDC[itdc] > 0) {
      printf(" %s:%d", ChannelNames[TDCSignal[itdc]].data(), mAssignedTDC[itdc]);
      if (mAssignedTDC[itdc] > 1) {
        hasMult = true;
      }
    }
  }
  if (hasMult) {
    printf(" MULTIPLE_ASSIGNMENTS\n");
  } else {
    printf("\n");
  }
#endif
  return 0;
} // reconstructTDC

int DigiReco::reconstruct(int ibeg, int iend)
{
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  LOG(info) << "________________________________________________________________________________";
  LOG(info) << __func__ << "(" << ibeg << ", " << iend << "): " << mReco[ibeg].ir.orbit << "." << mReco[ibeg].ir.bc << " - " << mReco[iend].ir.orbit << "." << mReco[iend].ir.bc;
#endif
  // Process consecutive BCs
  if (ibeg == iend) {
    mNLonely++;
    mLonely[mReco[ibeg].ir.bc]++;
    if (mBCData[ibeg].triggers != 0x0) {
      mLonelyTrig[mReco[ibeg].ir.bc]++;
    }
    // Cannot reconstruct lonely bunch
    // LOG(info) << "Lonely bunch " << mReco[ibeg].ir.orbit << "." << mReco[ibeg].ir.bc;
    return 0;
  }
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  for (int ibun = ibeg; ibun <= iend; ibun++) {
    printf("%d CH Mask: 0x%08x TDC data for:", ibun, mBCData[ibun].channels);
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
      if (mBCData[ibun].channels & mTDCMask[itdc]) {
        printf(" %s", ChannelNames[TDCSignal[itdc]].data());
      } else {
        printf("     ");
      }
    }
    printf("\n");
  }
#endif

  // After pile-up correction, find signals around main-main that satisfy condition on TDC
  findSignals(ibeg, iend);

  // For each calorimeter that has detects a collision at the time of main-main
  // collisions we reconstruct integrated charges and fill output tree
  for (int ich = 0; ich < NChannels; ich++) {
    // We consder the longest sequence of acquired bunches that can be readout by
    // the acquisition in case of isolated events (filling scheme with sigle bunches)
    uint32_t ref[NBCReadOut] = {ZDCRefInitVal, ZDCRefInitVal, ZDCRefInitVal, ZDCRefInitVal};
    // Flags to investigate pile-up
    bool hasFired[NBCReadOut] = {0}; // Channel has a TDC fired
    bool hasHit[NBCReadOut] = {0};   // Channel has hit
    bool hasAuto0[NBCReadOut] = {0}; // Module has Auto_0 trigger bit
    bool hasAutoM[NBCReadOut] = {0}; // Module has Auto_m trigger bit
    // Initialize info about previous bunches
    for (int ibp = 1; ibp < 4; ibp++) {
      int ibun = ibeg - ibp;
      // For the time being, we cannot access previous time frame (ibun<0)
      if (ibun < 0) {
        break;
      }
      auto bcd = mBCData[ibeg].ir.differenceInBC(mBCData[ibun].ir);
      if (bcd < 1) {
        LOG(error) << "Bunches are not in ascending order: " << mBCData[ibeg].ir.orbit << "." << mBCData[ibeg].ir.bc << " followed by " << mBCData[ibun].ir.orbit << "." << mBCData[ibun].ir.bc;
        return __LINE__;
      }
      if (bcd > 3) {
        break;
      }
      // Assignment is done taking into account bunch crossing difference
      ref[bcd] = mReco[ibun].ref[ich];
      // If channel has no waverform data cannot have a hit or trigger bit assigned
      // because all channels of a module are acquired if trigger condition is
      // satisfied
      if (ref[bcd] == ZDCRefInitVal) {
        continue;
      }
      // This condition is not comprehensive of all pile-up sources
      // chfired: Fired TDC condition related to channel (e.g. AND of TC and SUM @ main-main)
      hasFired[bcd] = mReco[ibun].chfired[ich];
      // Information from hardware autotrigger is not restricted to hits in main-main
      // but it may extend outside the correct bunch for signals near the bunch edges
      // hasHit refers to a single channel since it is derived from ChannelDataV0::Hit
      hasHit[bcd] = mBCData[ibun].triggers & mChMask[ich];
      // hasAuto0 and hasAutoM are derived from autotrigger decisions and therefore
      // refer to a module (max 4 channels) and not to a single channel
      ModuleTriggerMapData mt;
      mt.w = mBCData[ibun].moduleTriggers[mRopt->amod[ich]];
      hasAuto0[bcd] = mt.f.Auto_0;
      hasAutoM[bcd] = mt.f.Auto_m;
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
      printf("%2d %s bcd = %d ibun = %d ibeg = %d ref = %3u %s %s %s\n",
             ich, ChannelNames[ich].data(), bcd, ibun, ibeg, ref[bcd],
             hasHit[bcd] ? "H" : "-", hasAuto0[bcd] ? "A0" : "--", hasAutoM[bcd] ? "AM" : "--");
#endif
    }
    // Analyze all bunches
    for (int ibun = ibeg; ibun <= iend; ibun++) {
      updateOffsets(ibun); // Get Orbit pedestals
      auto& rec = mReco[ibun];
      // Check if the corresponding TDC is fired
      ref[0] = mReco[ibun].ref[ich];
      hasHit[0] = mBCData[ibun].triggers & mChMask[ich];
      ModuleTriggerMapData mt;
      mt.w = mBCData[ibun].moduleTriggers[mRopt->amod[ich]];
      hasAuto0[0] = mt.f.Auto_0;
      hasAutoM[0] = mt.f.Auto_m;
      if (rec.chfired[ich]) {
        // Check if channel data are present in payload
        if (ref[0] < ZDCRefInitVal) {
          // Energy reconstruction
          // Compute event by event pedestal
          bool hasEvPed = false;
          float evPed = 0;
          if (ibun > ibeg) {
            auto ref_m = ref[1];
            if (mRopt->beg_ped_int[ich] >= 0 || ref_m < ZDCRefInitVal) {
              for (int is = mRopt->beg_ped_int[ich]; is <= mRopt->end_ped_int[ich]; is++) {
                if (is < 0) {
                  // Sample is in previous BC
                  evPed += float(mChData[ref_m].data[is + NTimeBinsPerBC]);
                } else {
                  // Sample is in current BC
                  evPed += float(mChData[ref[0]].data[is]);
                }
              }
              evPed /= float(mRopt->end_ped_int[ich] - mRopt->beg_ped_int[ich] + 1);
              hasEvPed = true;
            }
          }
          // Pile-up detection using trigger information allows to identify
          // the presence of a signal in previous bunch and is module-wise

          // Detection of pile-up from previous bunch by comparing event pedestal with reference
          // (reference can be orbit or QC). If pile-up is detected we use orbit pedestal
          // instead of event pedestal
          // TODO: pedestal event could have a TM..
          if (hasEvPed && (mSource[ich] == PedOr || mSource[ich] == PedQC)) {
            auto pedref = mOffset[ich];
            if (evPed > pedref && (evPed - pedref) > mRopt->ped_thr_hi[ich]) {
              // Anomalous offset (put a warning but use event pedestal)
              rec.offPed[ich] = true;
            } else if (evPed < pedref && (pedref - evPed) > mRopt->ped_thr_lo[ich]) {
              // Possible presence of pile-up (will need to use orbit pedestal)
              rec.pilePed[ich] = true;
            }
          }
          float myPed = std::numeric_limits<float>::infinity();
          // TODO:
          if (hasEvPed && rec.pilePed[ich] == false) {
            myPed = evPed;
            rec.adcPedEv[ich] = true;
          } else if (mSource[ich] == PedOr) {
            myPed = mOffset[ich];
            rec.adcPedOr[ich] = true;
          } else if (mSource[ich] == PedQC) {
            myPed = mOffset[ich];
            rec.adcPedQC[ich] = true;
          } else {
            rec.adcPedMissing[ich] = true;
          }
          if (myPed < std::numeric_limits<float>::infinity()) {
            float sum = 0;
            for (int is = mRopt->beg_int[ich]; is <= mRopt->end_int[ich]; is++) {
              // TODO: pile-up correction
              // TODO: manage signal positioned across boundary
              sum += (myPed - float(mChData[ref[0]].data[is]));
            }
            rec.ezdc[ich] = sum * mRopt->energy_calib[ich];
          } else {
            LOGF(warn, "%d.%-4d CH %2d %s missing pedestal", rec.ir.orbit, rec.ir.bc, ich, ChannelNames[ich].data());
          }
        } else {
          // This could arise from memory corruption or in the case when TDC bits are forced to 1
          // due to a broken channel. For example Sum is broken and sum TDC is forced to 1
          // Take a very small signal that triggers on one module and not the other
          // 1) module 0 triggered by ZNATC is autotriggered
          // 2) module 1 triggered by ZNATC is not triggered -> Sum is missing
          // 3) sum TDC is forced to 1 and therefore the TDC is fired even if data are not present
          // 4) you get this error flag in reconstruction
          // This should happen only in case of hardware fault and signals near threshold
          // This should be mitigated by having a software threshold higher than the hardware one
          rec.adcPedMissing[ich] = true;
          if (mVerbosity >= DbgMedium) {
            LOGF(warn, "%d.%-4d CH %2d %s ADC missing, TDC present", rec.ir.orbit, rec.ir.bc, ich, ChannelNames[ich].data());
          }
        }
      }
      // Shift bunches by 1 taking into account bunch crossing difference
      // TODO
      // This is a simple shift by 1 position
      if (ibun != iend) {
        for (int ibcr = NBCReadOut - 1; ibcr > 0; ibcr--) {
          ref[ibcr] = ref[ibcr - 1];
          hasHit[ibcr] = hasHit[ibcr - 1];
          hasAuto0[ibcr] = hasAuto0[ibcr - 1];
          hasAutoM[ibcr] = hasAutoM[ibcr - 1];
          hasFired[ibcr] = hasFired[ibcr - 1];
        }
      }
    } // Loop on bunches
  }   // Loop on channels
  if (mTreeDbg) {
    for (int ibun = ibeg; ibun <= iend; ibun++) {
      auto& rec = mReco[ibun];
      mRec = rec;
      mTDbg->Fill();
    }
  }
  return 0;
} // reconstruct

void DigiReco::updateOffsets(int ibun)
{
  auto orbit = mBCData[ibun].ir.orbit;
  if (orbit == mOffsetOrbit) {
    return;
  }
  mOffsetOrbit = orbit;

  // Reset information about pedestal origin
  for (int ich = 0; ich < NChannels; ich++) {
    mSource[ich] = PedND;
    mOffset[ich] = std::numeric_limits<float>::infinity();
  }

  // Default TDC pedestal is from orbit
  // Look for Orbit pedestal offset
  std::map<uint32_t, int>::iterator it = mOrbit.find(orbit);
  if (it != mOrbit.end()) {
    auto& orbitdata = mOrbitData[it->second];
    for (int ich = 0; ich < NChannels; ich++) {
      auto myped = float(orbitdata.data[ich]) * mModuleConfig->baselineFactor;
      if (myped >= ADCMin && myped <= ADCMax) {
        // Pedestal information is present for this channel
        mOffset[ich] = myped;
        mSource[ich] = PedOr;
      }
    }
  }

  // Use average "QC" pedestal if orbit pedestals are missing
  if (mPedParam != nullptr) {
    for (int ich = 0; ich < NChannels; ich++) {
      if (mSource[ich] == PedND) {
        auto myped = mPedParam->getCalib(ich);
        if (myped >= ADCMin && myped <= ADCMax) {
          mOffset[ich] = myped;
          mSource[ich] = PedQC;
        }
      }
    }
  }

  for (int ich = 0; ich < NChannels; ich++) {
    if (mSource[ich] == PedND) {
      mMissingPed[ich]++;
      if (mVerbosity > DbgMinimal) {
        LOGF(error, "Missing pedestal for ch %2d %s orbit %u ", ich, ChannelNames[ich], mOffsetOrbit);
      }
    }
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
    LOGF(info, "Pedestal for ch %2d %s orbit %u %s: %f", ich, ChannelNames[ich], mOffsetOrbit, mSource[ich] == PedOr ? "OR" : (mSource[ich] == PedQC ? "QC" : "??"), mOffset[ich]);
#endif
  }
} // updateOffsets

int DigiReco::processTrigger(int itdc, int ibeg, int iend)
{
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  LOG(info) << __func__ << "(itdc=" << itdc << "[" << ChannelNames[TDCSignal[itdc]] << "], " << ibeg << ", " << iend << "): " << mReco[ibeg].ir.orbit << "." << mReco[ibeg].ir.bc << " - " << mReco[iend].ir.orbit << "." << mReco[iend].ir.bc;
#endif
  // Extracting TDC information for TDC number itdc, in consecutive bunches from ibeg to iend
  int nbun = iend - ibeg + 1;
  int maxs2 = NTimeBinsPerBC * nbun - 1;
  int shift = mRopt->tsh[itdc];
  int thr = mRopt->tth[itdc];

  int is1 = 0, is2 = 1;
  uint8_t isfired = 0;
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  int16_t m[3] = {0};
  int16_t s[3] = {0};
#endif
  int it1 = 0, it2 = 0, ib1 = -1, ib2 = -1;
  for (;;) {
    // Shift data
    isfired = isfired << 1;
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
    for (int i = 2; i > 0; i--) {
      m[i] = m[i - 1];
      s[i] = s[i - 1];
    }
#endif
    // Bunches and samples that are used in the difference
    int b1 = ibeg + is1 / NTimeBinsPerBC;
    int b2 = ibeg + is2 / NTimeBinsPerBC;
    int s1 = is1 % NTimeBinsPerBC;
    int s2 = is2 % NTimeBinsPerBC;
    auto ref_m = mReco[b1].ref[TDCSignal[itdc]]; // reference to minuend
    auto ref_s = mReco[b2].ref[TDCSignal[itdc]]; // reference to subtrahend
    // Check data consistency before computing difference
    if (ref_m == ZDCRefInitVal || ref_s == ZDCRefInitVal) {
      if (ref_m == ZDCRefInitVal) {
        LOG(error) << __func__ << " @ " << __LINE__ << " Missing information for bunch crossing " << mReco[b1].ir.orbit << "." << mReco[b1].ir.bc << " tdc = " << itdc << " sig = " << TDCSignal[itdc];
      }
      if (ref_s == ZDCRefInitVal) {
        LOG(error) << __func__ << " @ " << __LINE__ << " Missing information for bunch crossing " << mReco[b2].ir.orbit << "." << mReco[b2].ir.bc << " tdc = " << itdc << " sig = " << TDCSignal[itdc];
      }
      return __LINE__;
    }
    // Check that bunch crossings are indeed the same or consecutive
    auto bcd = mReco[b2].ir.differenceInBC(mReco[b1].ir);
    if (bcd != 0 && bcd != 1) {
      LOG(error) << __func__ << " @ " << __LINE__ << " Large bunch crossing difference " << mReco[b1].ir.orbit << "." << mReco[b1].ir.bc << " followed by " << mReco[b2].ir.orbit << "." << mReco[b2].ir.bc;
      return __LINE__;
    }
    int diff = mChData[ref_m].data[s1] - mChData[ref_s].data[s2];
    // Triple trigger condition
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
    m[0] = mChData[ref_m].data[s1];
    s[0] = mChData[ref_s].data[s2];
#endif
    if (diff > thr) {
      isfired = isfired | 0x1;
      if ((isfired & mTriggerCondition) == mTriggerCondition) {
        // Fired bit is assigned to the second sample, i.e. to the one that can identify the
        // signal peak position
        mReco[b2].fired[itdc] |= mMask[s2];
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
        if (mTriggerCondition == 0x7) {
          printf("0x7 TDC %d[%s] Fired @ %u.%u.s%02u (%5d-%5d)=%5d>%2d && (%5d-%5d)=%5d>%2d && (s%02d:%-5d-s%02d:%-5d)=%5d>%2d\n",
                 itdc, ChannelNames[TDCSignal[itdc]].data(), mReco[b2].ir.orbit, mReco[b2].ir.bc, s2,
                 m[2], s[2], (m[2] - s[2]), thr,
                 m[1], s[1], (m[1] - s[1]), thr,
                 s1, m[0], s2, s[0], diff, thr);
        } else if (mTriggerCondition == 0x3) {
          printf("0x3 TDC %d[%s] Fired @ %u.%u.s%02u (%5d-%5d)=%5d>%2d && (s%02d:%-5d-s%02d:(%-5d))=%5d>%2d\n",
                 itdc, ChannelNames[TDCSignal[itdc]].data(), mReco[b2].ir.orbit, mReco[b2].ir.bc, s2,
                 m[1], s[1], (m[1] - s[1]), thr,
                 s1, m[0], s2, s[0], diff, thr);
        } else if (mTriggerCondition == 0x1) {
          printf("0x1 TDC %d[%s] Fired @ %u.%u.s%02u (%d-(%d))=%d>%d && (%d-(%d))=%d>%d && (s%d:%d-s%d:(%d))=%d>%d\n",
                 itdc, ChannelNames[TDCSignal[itdc]].data(), mReco[b2].ir.orbit, mReco[b2].ir.bc, s2,
                 s1, m[0], s2, s[0], diff, thr);
        }
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
  return interpolate(itdc, ibeg, iend);
} // processTrigger

int DigiReco::processTriggerExtended(int itdc, int ibeg, int iend)
{
  auto isig = TDCSignal[itdc];
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  LOG(info) << __func__ << "(itdc=" << itdc << "[" << ChannelNames[isig] << "], " << ibeg << ", " << iend << "): " << mReco[ibeg].ir.orbit << "." << mReco[ibeg].ir.bc << " - " << mReco[iend].ir.orbit << "." << mReco[iend].ir.bc;
#endif
  // Extends search zone at the beginning of sequence. Need pedestal information.
  // For simplicity we use information for current bunch/orbit
  updateOffsets(ibeg);
  if (mSource[isig] == PedND) {
    // Fall back to normal trigger
    // Message will be produced when computing amplitude (if a hit is found in this bunch)
    // In this framework we have a potential undetected inefficiency, however pedestal
    // problem is a serious problem and will be noticed anyway
    return processTrigger(itdc, ibeg, iend);
  }

  int nbun = iend - ibeg + 1;
  int maxs2 = NTimeBinsPerBC * nbun - 1;
  int shift = mRopt->tsh[itdc];
  int thr = mRopt->tth[itdc];

  int is1 = -shift, is2 = 0;
  uint8_t isfired = 0;
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  int16_t m[3] = {0};
  int16_t s[3] = {0};
#endif

  int it1 = 0, it2 = 0, ib1 = -1, ib2 = -1;
  for (;;) {
    // Shift data
    isfired = isfired << 1;
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
    for (int i = 2; i > 0; i--) {
      m[i] = m[i - 1];
      s[i] = s[i - 1];
    }
#endif
    // Bunches and samples that are used in the difference
    int diff = 0;
    int b2 = ibeg + is2 / NTimeBinsPerBC;
    int s2 = is2 % NTimeBinsPerBC;
    auto ref_s = mReco[b2].ref[isig]; // reference to subtrahend
    int s1 = is1 % NTimeBinsPerBC;
    if (is1 < 0) {
      if (ref_s == ZDCRefInitVal) {
        LOG(error) << __func__ << " @ " << __LINE__ << " Missing information for bunch crossing " << mReco[b2].ir.orbit << "." << mReco[b2].ir.bc << " sig = " << isig;
        return __LINE__;
      }
      diff = mOffset[isig] - mChData[ref_s].data[s2];
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
      m[0] = mOffset[isig];
      s[0] = mChData[ref_s].data[s2];
#endif
    } else {
      int b1 = ibeg + is1 / NTimeBinsPerBC;
      auto ref_m = mReco[b1].ref[TDCSignal[itdc]]; // reference to minuend
      // Check data consistency before computing difference
      if (ref_m == ZDCRefInitVal || ref_s == ZDCRefInitVal) {
        LOG(error) << __func__ << " @ " << __LINE__ << " Missing information for bunch crossing " << mReco[b1].ir.orbit << "." << mReco[b1].ir.bc << " tdc = " << itdc << " sig = " << TDCSignal[itdc];
        return __LINE__;
      }
      // Check that bunch crossings are indeed the same or consecutive
      auto bcd = mReco[b2].ir.differenceInBC(mReco[b1].ir);
      if (bcd != 0 && bcd != 1) {
        LOG(error) << __func__ << ": large bunch crossing difference " << mReco[b1].ir.orbit << "." << mReco[b1].ir.bc << " followed by " << mReco[b2].ir.orbit << "." << mReco[b2].ir.bc;
        return __LINE__;
      }
      diff = mChData[ref_m].data[s1] - mChData[ref_s].data[s2];
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
      m[0] = mChData[ref_m].data[s1];
      s[0] = mChData[ref_s].data[s2];
#endif
    }
    if (diff > thr) {
      isfired = isfired | 0x1;
      // Check trigger condition
      if ((isfired & mTriggerCondition) == mTriggerCondition) {
        // Fired bit is assigned to the second sample, i.e. to the one that can identify the
        // signal peak position
        mReco[b2].fired[itdc] |= mMask[s2];
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
        if (mTriggerCondition == 0x7) {
          printf("0x7E TDC %d[%s] Fired @ %u.%u.s%02u (%5d-%5d)=%5d>%2d && (%5d-%5d)=%5d>%5d && (s%02d:%-5d-s%02d:%-5d)=%-5d>%2d\n",
                 itdc, ChannelNames[TDCSignal[itdc]].data(), mReco[b2].ir.orbit, mReco[b2].ir.bc, s2,
                 m[2], s[2], (m[2] - s[2]), thr,
                 m[1], s[1], (m[1] - s[1]), thr,
                 s1, m[0], s2, s[0], diff, thr);
        } else if (mTriggerCondition == 0x3) {
          printf("0x3E TDC %d[%s] Fired @ %u.%u.s%02u (%5d-%5d)=%5d>%2d && (s%02d:%-5d-s%02d:(%-5d))=%-5d>%2d\n",
                 itdc, ChannelNames[TDCSignal[itdc]].data(), mReco[b2].ir.orbit, mReco[b2].ir.bc, s2,
                 m[1], s[1], (m[1] - s[1]), thr,
                 s1, m[0], s2, s[0], diff, thr);
        } else if (mTriggerCondition == 0x1) {
          printf("0x1E TDC %d[%s] Fired @ %u.%u.s%02u (%5d-(%5d))=%5d>%2d && (%5d-(%5d))=%5d>%2d && (s%d:%5d-s%d:(%5d))=%5d>%2d\n",
                 itdc, ChannelNames[TDCSignal[itdc]].data(), mReco[b2].ir.orbit, mReco[b2].ir.bc, s2,
                 s1, m[0], s2, s[0], diff, thr);
        }
#endif
      }
    }
    is1++;
    if (is2 < maxs2) {
      is2++;
    }
    if (is1 == maxs2) {
      break;
    }
  }
  return interpolate(itdc, ibeg, iend);
} // processTriggerExtended

// Interpolation for single point
O2_ZDC_DIGIRECO_FLT DigiReco::getPoint(int isig, int ibeg, int iend, int i)
{
  constexpr int nsbun = TSN * NTimeBinsPerBC; // Total number of interpolated points per bunch crossing
  if (i >= mNtot || i < 0) {
    LOG(error) << "Error addressing isig=" << isig << " i=" << i << " mNtot=" << mNtot;
    mInError = true;
    return std::numeric_limits<float>::infinity();
  }
  // Constant extrapolation at the beginning and at the end of the array
  if (i < TSNH) {
    // Return value of first sample
    return mFirstSample;
  } else if (i >= mIlast) {
    // Return value of last sample
    return mLastSample;
  } else {
    // Identification of the point to be assigned
    int ibun = ibeg + i / nsbun;
    // Interpolation between acquired points (N.B. from 0 to mNint)
    i = i - TSNH;
    int im = i % TSN;
    if (im == 0) {
      // This is an acquired point
      int ip = (i / TSN) % NTimeBinsPerBC;
      int ib = ibeg + (i / TSN) / NTimeBinsPerBC;
      if (ib != ibun) {
        LOG(error) << "ib=" << ib << " ibun=" << ibun;
        mInError = true;
        return std::numeric_limits<float>::infinity();
      }
      return mReco[ibun].data[isig][ip]; // Filtered point
      // return mChData[mReco[ibun].ref[isig]].data[ip]; // Original point
    } else {
      // Do the actual interpolation
      O2_ZDC_DIGIRECO_FLT y = 0;
      int ip = i / TSN;
      O2_ZDC_DIGIRECO_FLT sum = 0;
      for (int is = TSN - im, ii = ip - TSL + 1; is < NTS; is += TSN, ii++) {
        // Default is first point in the array
        O2_ZDC_DIGIRECO_FLT yy = mFirstSample;
        if (ii > 0) {
          if (ii < mNsam) {
            int ip = ii % NTimeBinsPerBC;
            int ib = ibeg + ii / NTimeBinsPerBC;
            yy = mReco[ib].data[isig][ip];
            // yy = mChData[mReco[ib].ref[isig]].data[ip];
          } else {
            // Last acquired point
            yy = mLastSample;
          }
        }
        sum += mTS[is];
        y += yy * mTS[is];
      }
      y = y / sum;
      return y;
    }
  }
}

void DigiReco::setPoint(int isig, int ibeg, int iend, int i)
{
  // This function needs to be used only if mFullInterpolation is true otherwise the
  // vectors are not allocated
  if (!mFullInterpolation) {
    LOG(fatal) << __func__ << " call with mFullInterpolation = " << mFullInterpolation;
    return;
  }
  constexpr int nsbun = TSN * NTimeBinsPerBC; // Total number of interpolated points per bunch crossing
  if (i >= mNtot || i < 0) {
    LOG(error) << "Error addressing signal isig=" << isig << " i=" << i << " mNtot=" << mNtot;
    mInError = true;
    return;
  }
  // Constant extrapolation at the beginning and at the end of the array
  if (i < TSNH) {
    // Assign value of first sample
    mReco[ibeg].inter[isig][i] = mFirstSample;
  } else if (i >= mIlast) {
    // Assign value of last sample
    int isam = i % nsbun;
    mReco[iend].inter[isig][isam] = mLastSample;
  } else {
    // Identification of the point to be assigned
    int ibun = ibeg + i / nsbun;
    int isam = i % nsbun;
    mReco[ibun].inter[isig][isam] = getPoint(isig, ibeg, iend, i);
  }
} // setPoint

int DigiReco::fullInterpolation(int isig, int ibeg, int iend)
{
  // Interpolation of signal isig, in consecutive bunches from ibeg to iend
  // This function works for all signals and does not evaluate trigger
  // You need to call interpolate(int itdc... for the TDC signals
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  LOG(info) << __func__ << "(isig=" << isig << "[" << ChannelNames[isig] << "], " << ibeg << ", " << iend << "): " << mReco[ibeg].ir.orbit << "." << mReco[ibeg].ir.bc << " - " << mReco[iend].ir.orbit << "." << mReco[iend].ir.bc;
#endif

  // TODO: get data from preceding time frame in case there are bunches
  // with signal at the beginning of the first orbit of a time frame

  constexpr int MaxTimeBin = NTimeBinsPerBC - 1; //< number of samples per BC

  // Set data members for interpolation of the current channel
  mNbun = iend - ibeg + 1;                    // Number of adjacent bunches
  mNsam = mNbun * NTimeBinsPerBC;             // Number of acquired samples
  mNtot = mNsam * TSN;                        // Total number of points in the interpolated arrays
  mNint = (mNbun * NTimeBinsPerBC - 1) * TSN; // Total points in the interpolation region (-1)
  mIlast = mNtot - TSNH;                      // Index of last acquired sample

  // At this level there should be no need to check if the channel is connected
  // since a fatal should have been raised already
  for (int ibun = ibeg; ibun <= iend; ibun++) {
    auto ref = mReco[ibun].ref[isig];
    if (ref == ZDCRefInitVal) {
      LOG(error) << __func__ << " @ " << __LINE__ << " Missing information for bunch crossing " << mReco[ibun].ir.orbit << "." << mReco[ibun].ir.bc << " sig = " << isig;
      return __LINE__;
    }
  }

  mFirstSample = mReco[ibeg].data[isig][0];
  mLastSample = mReco[iend].data[isig][MaxTimeBin];

  // Allocate and fill array of interpolated points
  for (int ibun = ibeg; ibun <= iend; ibun++) {
    mReco[ibun].allocate(isig);
  }
  for (int i = 0; i < mNtot; i++) {
    setPoint(isig, ibeg, iend, i);
  }
  if (mInError) {
    return __LINE__;
  }
  return 0;
}

int DigiReco::interpolate(int itdc, int ibeg, int iend)
{
  // Interpolation of TDC channel itdc, in consecutive bunches from ibeg to iend
  int isig = TDCSignal[itdc];

#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  LOG(info) << __func__ << "(itdc=" << itdc << "[" << ChannelNames[isig] << "], " << ibeg << ", " << iend << "): " << mReco[ibeg].ir.orbit << "." << mReco[ibeg].ir.bc << " - " << mReco[iend].ir.orbit << "." << mReco[iend].ir.bc;
#endif

  // TODO: get data from preceding time frame in case there are bunches
  // with signal at the beginning of the first orbit of a time frame

  constexpr int MaxTimeBin = NTimeBinsPerBC - 1; //< number of samples per BC
  constexpr int nsbun = TSN * NTimeBinsPerBC;    // Total number of interpolated points per bunch crossing

  // Set data members for interpolation of the current TDC
  mNbun = iend - ibeg + 1;                    // Number of adjacent bunches
  mNsam = mNbun * NTimeBinsPerBC;             // Number of acquired samples
  mNtot = mNsam * TSN;                        // Total number of points in the interpolated arrays
  mNint = (mNbun * NTimeBinsPerBC - 1) * TSN; // Total points in the interpolation region (-1)
  mIlast = mNtot - TSNH;                      // Index of last acquired sample

  constexpr int nsp = 5; // Number of points to be searched

  // At this level there should be no need to check if the channel is connected
  // since a fatal should have been raised already
  for (int ibun = ibeg; ibun <= iend; ibun++) {
    auto ref = mReco[ibun].ref[isig];
    if (ref == ZDCRefInitVal) {
      LOG(error) << __func__ << " @ " << __LINE__ << " Missing information for bunch crossing " << mReco[ibun].ir.orbit << "." << mReco[ibun].ir.bc << " sig = " << isig;
      return __LINE__;
    }
  }

  // auto ref_beg = mReco[ibeg].ref[isig];
  // auto ref_end = mReco[iend].ref[isig];
  // mFirstSample = mChData[ref_beg].data[0]; // Original points
  // mLastSample = mChData[ref_end].data[MaxTimeBin]; // Original points

  mFirstSample = mReco[ibeg].data[isig][0];
  mLastSample = mReco[iend].data[isig][MaxTimeBin];

  // mFullInterpolation turns on full interpolation for debugging
  // otherwise the interpolation is performed only around actual signal
  if (mFullInterpolation) {
    for (int ibun = ibeg; ibun <= iend; ibun++) {
      mReco[ibun].allocate(isig);
    }
    for (int i = 0; i < mNtot; i++) {
      setPoint(isig, ibeg, iend, i);
    }
  }
  if (mInError) {
    return __LINE__;
  }
  // Looking for a local maximum in a search zone
  O2_ZDC_DIGIRECO_FLT amp = std::numeric_limits<float>::infinity(); // Amplitude to be stored
  int isam_amp = 0;                                                 // Sample at maximum amplitude (relative to beginning of group)
  int ip_old = -1, ip_cur = -1, ib_cur = -1;                        // Current and old points
  bool is_searchable = false;                                       // Flag for point in the search zone for maximum amplitude
  bool was_searchable = false;                                      // Flag for point in the search zone for maximum amplitude
  int ib[nsp] = {-1, -1, -1, -1, -1};
  int ip[nsp] = {-1, -1, -1, -1, -1};
  // N.B. Points at the extremes are constant therefore no local maximum
  // can occur in these two regions
  for (int i = 0; i < mNint; i += mInterpolationStep) {
    int isam = i + TSNH;
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
    // with satellites
    // This is buggy because you can get just one TDC for each search zone
    // If more than one signal is present, just the largest one is saved
    // To overcome this limitation one should implement more refined analysis
    // techniques to perform shape recognition

    // If we exit from searching zone
    if (was_searchable && !is_searchable) {
      if (amp <= ADCMax) {
        if (mInterpolationStep > 1) {
          // Refine peak search
          int sbeg = (isam_amp - TSNH);
          int send = (isam_amp + TSNH);
          if (sbeg < 0) {
            sbeg = 0;
            send = sbeg + TSN;
          }
          if (send > (mNint + TSNH)) {
            send = mNint + TSNH;
            sbeg = send - TSN;
          }
          if (sbeg < 0) {
            sbeg = 0;
          }
          for (int spos = sbeg; spos < send; spos++) {
            // Perform interpolation for the searched point
            O2_ZDC_DIGIRECO_FLT myval = getPoint(isig, ibeg, iend, spos);
            // Get local minimum of waveform
            if (myval < amp) {
              amp = myval;
              isam_amp = spos;
            }
          }
        }
        // Store identified peak
        int ibun = ibeg + isam_amp / nsbun;
        updateOffsets(ibun);
        // At this level offsets are from Orbit or QC therefore
        // the TDC amplitude and time are affected by pile-up from
        // previous collisions. Pile up correction needs to be
        // performed after all signals have been identified
        if (mSource[isig] != PedND) {
          amp = mOffset[isig] - amp;
        } else {
          LOGF(error, "%u.%-4d Missing pedestal for TDC %d %s ", mBCData[ibun].ir.orbit, mBCData[ibun].ir.bc, itdc, ChannelNames[TDCSignal[itdc]]);
          amp = std::numeric_limits<float>::infinity();
        }
        int tdc = isam_amp % nsbun;
        assignTDC(ibun, ibeg, iend, itdc, tdc, amp);
      }
      amp = std::numeric_limits<float>::infinity();
      isam_amp = 0;
      was_searchable = 0;
    }
    if (is_searchable) {
      int mysam = isam % nsbun;
      O2_ZDC_DIGIRECO_FLT myval;
      if (mFullInterpolation) {
        // Already interpolated
        myval = mReco[ib_cur].inter[isig][mysam];
      } else {
        // Perform interpolation for the searched point
        myval = getPoint(isig, ibeg, iend, isam);
      }
      // Get local minimum of waveform
      if (myval < amp) {
        amp = myval;
        isam_amp = isam;
      }
    }
  } // Loop on interpolated points
  if (mInError) {
    return __LINE__;
  }

  // Trigger flag still present at the of the scan
  if (is_searchable) {
    // Add last identified peak
    if (amp <= ADCMax) {
      if (mInterpolationStep > 1) {
        // Refine peak search
        int sbeg = (isam_amp - TSNH);
        int send = (isam_amp + TSNH);
        if (sbeg < 0) {
          sbeg = 0;
          send = sbeg + TSN;
        }
        if (send > (mNint + TSNH)) {
          send = mNint + TSNH;
          sbeg = send - TSN;
        }
        if (sbeg < 0) {
          sbeg = 0;
        }
        for (int spos = sbeg; spos < send; spos++) {
          // Perform interpolation for the searched point
          O2_ZDC_DIGIRECO_FLT myval = getPoint(isig, ibeg, iend, spos);
          // Get local minimum of waveform
          if (myval < amp) {
            amp = myval;
            isam_amp = spos;
          }
        }
      }
      // Store identified peak
      int ibun = ibeg + isam_amp / nsbun;
      updateOffsets(ibun);
      if (mSource[isig] != PedND) {
        amp = mOffset[isig] - amp;
      } else {
        LOGF(error, "%u.%-4d Missing pedestal for TDC %d %s ", mBCData[ibun].ir.orbit, mBCData[ibun].ir.bc, itdc, ChannelNames[TDCSignal[itdc]]);
        amp = std::numeric_limits<float>::infinity();
      }
      int tdc = isam_amp % nsbun;
      assignTDC(ibun, ibeg, iend, itdc, tdc, amp);
    }
  }
  if (mInError) {
    return __LINE__;
  }
  // TODO: add logic to assign TDC in presence of overflow
  return 0;
} // interpolate

void DigiReco::assignTDC(int ibun, int ibeg, int iend, int itdc, int tdc, float amp)
{
  constexpr int nsbun = TSN * NTimeBinsPerBC; // Total number of interpolated points per bunch crossing
  constexpr int tdc_max = nsbun / 2;
  constexpr int tdc_min = -tdc_max;

  auto& rec = mReco[ibun];

  // Flag hit position in sequence
  if (ibun == ibeg) {
    rec.isBeg[itdc] = true;
  }
  if (ibun == iend) {
    rec.isEnd[itdc] = true;
  }

  int isig = TDCSignal[itdc];
  float TDCVal = tdc;
  float TDCAmp = amp;

  // Correct for time bias on single signals
  float TDCValCorr = 0, TDCAmpCorr = 0;
  if (mCorrSignal == false || correctTDCSignal(itdc, tdc, amp, TDCValCorr, TDCAmpCorr, rec.isBeg[itdc], rec.isEnd[itdc]) != 0) {
    // Cannot apply amplitude correction for isolated signal -> Flag error condition
    rec.tdcSigE[TDCSignal[itdc]] = true;
  } else {
    TDCVal = TDCValCorr;
    // Cannot correct amplitude if pedestal is missing
    if (!rec.tdcPedMissing[isig]) {
      TDCAmp = TDCAmpCorr;
    }
  }

  // rec.genericE[TDCSignal[itdc]] = true; // Produce fake error

  // TDC calibration
  TDCVal = TDCVal - tdc_shift[itdc];
  if (!rec.tdcPedMissing[isig]) {
    // Cannot correct amplitude if pedestal is missing
    TDCAmp = TDCAmp * tdc_calib[itdc];
  }

  // Encode amplitude and assign
  auto myamp = TDCAmp / o2::zdc::FTDCAmp;
  rec.TDCVal[itdc].push_back(TDCVal);
  rec.TDCAmp[itdc].push_back(myamp);
  int& ihit = mReco[ibun].ntdc[itdc];
#ifdef O2_ZDC_TDC_C_ARRAY
  if (ihit < MaxTDCValues) {
    rec.tdcVal[itdc][ihit] = TDCVal;
    rec.tdcAmp[itdc][ihit] = myamp;
  } else {
    LOG(error) << rec.ir.orbit << "." << rec.ir.bc << " ibun=" << ibun << " itdc=" << itdc << " tdc=" << tdc << " TDCVal=" << TDCVal * o2::zdc::FTDCVal << " TDCAmp=" << TDCAmp * o2::zdc::FTDCAmp << " OVERFLOW";
  }
#endif
  // Assign info about pedestal subtration
  if (mSource[isig] == PedOr) {
    rec.tdcPedOr[isig] = true;
  } else if (mSource[isig] == PedQC) {
    rec.tdcPedQC[isig] = true;
  } else if (mSource[isig] == PedEv) {
    // In present implementation this never happens
    rec.tdcPedEv[isig] = true;
  } else {
    rec.tdcPedMissing[isig] = true;
  }
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  LOG(info) << __func__ << " itdc=" << itdc << " " << ChannelNames[isig] << " @ ibun=" << ibun << " " << mReco[ibun].ir.orbit << "." << mReco[ibun].ir.bc << " "
            << " tdc=" << tdc << " -> " << TDCValCorr << " shift=" << tdc_shift[itdc] << " -> TDCVal=" << TDCVal << "=" << TDCVal * o2::zdc::FTDCVal
            << " mSource[" << isig << "] = " << unsigned(mSource[isig]) << " = " << mOffset[isig]
            << " amp=" << amp << " -> " << TDCAmpCorr << " calib=" << tdc_calib[itdc] << " -> TDCAmp=" << TDCAmp << "=" << myamp
            << (ibun == ibeg ? " B" : "") << (ibun == iend ? " E" : "");
  mAssignedTDC[itdc]++;
#endif
  ihit++;
} // assignTDC

void DigiReco::findSignals(int ibeg, int iend)
{
  // N.B. findSignals is called after pile-up correction on TDCs
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  LOG(info) << __func__ << "(" << ibeg << ", " << iend << "): " << mReco[ibeg].ir.orbit << "." << mReco[ibeg].ir.bc << " - " << mReco[iend].ir.orbit << "." << mReco[iend].ir.bc;
#endif
  // Identify TDC signals
  for (int ibun = ibeg; ibun <= iend; ibun++) {
    updateOffsets(ibun); // Get orbit pedestals or run pedestals as a fallback
    auto& rec = mReco[ibun];
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
      bool msg = false;
      if (rec.fired[itdc] != 0x0) {
        msg = true;
        printf("%d %u.%-4u TDCDiscr %d [%s] 0x%04hx -> ", ibun, rec.ir.orbit, rec.ir.bc, itdc, ChannelNames[TDCSignal[itdc]].data(), rec.fired[itdc]);
        for (int isam = 0; isam < NTimeBinsPerBC; isam++) {
          printf("%d", rec.fired[itdc] & mMask[isam] ? 1 : 0);
        }
      }
#endif
      rec.pattern[itdc] = 0;
      for (int32_t i = 0; i < rec.ntdc[itdc]; i++) {
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
        msg = true;
        printf(" %d TDC A=%5.0f @ T=%5.0f", i, rec.TDCAmp[itdc][i], rec.TDCVal[itdc][i]);
#endif
        // There is a TDC value in the search zone around main-main position
        // NB: by definition main-main collision has zero time
        if (std::abs(rec.TDCVal[itdc][i]) < mRopt->tdc_search[itdc]) {
          rec.pattern[itdc] = 1;
        }
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
        if (rec.pattern[itdc] == 1) {
          printf("  in_r");
        } else {
          printf(" out_r");
        }
#endif
      }
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
      if (msg) {
        printf("\n");
      }
#endif
    }

#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
    printf("%d %u.%-4u TDC PATTERN: ", ibun, mReco[ibun].ir.orbit, mReco[ibun].ir.bc);
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
      printf("%d", rec.pattern[itdc]);
    }
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
      printf(" %s", rec.pattern[itdc] ? ChannelNames[TDCSignal[itdc]].data() : "     ");
    }
    printf("\n");
#endif

    // Check if coincidence of common PM and sum of towers is satisfied
    // Missing coincidence can be due for example by PM noise
    // TODO: for the moment the map of fired TDC is in RecEventAux
    // One should  transmit it or introduce a way to recompute while reading back
    // the reconstructed events
    // Side A
    if ((rec.pattern[TDCZNAC] || mRopt->bitset[TDCZNAC]) && (rec.pattern[TDCZNAS] || mRopt->bitset[TDCZNAS])) {
      for (int ich = IdZNAC; ich <= IdZNASum; ich++) {
        rec.chfired[ich] = true;
      }
    }
    if ((rec.pattern[TDCZPAC] || mRopt->bitset[TDCZPAC]) && (rec.pattern[TDCZPAS] || mRopt->bitset[TDCZPAS])) {
      for (int ich = IdZPAC; ich <= IdZPASum; ich++) {
        rec.chfired[ich] = true;
      }
    }
    // ZEM1 and ZEM2 are not in coincidence
    rec.chfired[IdZEM1] = rec.pattern[TDCZEM1];
    rec.chfired[IdZEM2] = rec.pattern[TDCZEM2];
    // Side C
    if ((rec.pattern[TDCZNCC] || mRopt->bitset[TDCZNCC]) && (rec.pattern[TDCZNCS] || mRopt->bitset[TDCZNCS])) {
      for (int ich = IdZNCC; ich <= IdZNCSum; ich++) {
        rec.chfired[ich] = true;
      }
    }
    if ((rec.pattern[TDCZPCC] || mRopt->bitset[TDCZPCC]) && (rec.pattern[TDCZPCS] || mRopt->bitset[TDCZPCS])) {
      for (int ich = IdZPCC; ich <= IdZPCSum; ich++) {
        rec.chfired[ich] = true;
      }
    }
    // TODO: option to use a less restrictive definition of "fired" using for example
    // just the autotrigger bits instead of using TDCs
#ifndef O2_ZDC_DEBUG
    if (mVerbosity >= DbgFull) {
      printf("%d %u.%-4u TDC FIRED ", ibun, rec.ir.orbit, rec.ir.bc);
      printf("ZNA:%d%d%d%d%d%d ZPA:%d%d%d%d%d%d ZEM:%d%d ZNC:%d%d%d%d%d%d ZPC:%d%d%d%d%d%d\n",
             rec.chfired[IdZNAC], rec.chfired[IdZNA1], rec.chfired[IdZNA2], rec.chfired[IdZNA3], rec.chfired[IdZNA4], rec.chfired[IdZNASum],
             rec.chfired[IdZPAC], rec.chfired[IdZPA1], rec.chfired[IdZPA2], rec.chfired[IdZPA3], rec.chfired[IdZPA4], rec.chfired[IdZPASum],
             rec.chfired[IdZEM1], rec.chfired[IdZEM2],
             rec.chfired[IdZNCC], rec.chfired[IdZNC1], rec.chfired[IdZNC2], rec.chfired[IdZNC3], rec.chfired[IdZNC4], rec.chfired[IdZNCSum],
             rec.chfired[IdZPCC], rec.chfired[IdZPC1], rec.chfired[IdZPC2], rec.chfired[IdZPC3], rec.chfired[IdZPC4], rec.chfired[IdZPCSum]);
    }
#endif
  } // loop on bunches
} // findSignals

void DigiReco::correctTDCPile()
{
  // Pile-up correction for TDCs
  // FEE acquires data in two modes: triggered and continuous
  // In triggered mode minimum two and up to four bunches are transferred for each ALICE trigger
  // In continuous mode two bunches are transferred for each autotrigger
  // Reconstruction is performed for consecutive bunches
  // There is no issue with gaps for triggered mode because the trigger condition
  // A0 || A1 || (A2 && (T0 || TM)) || (A3 && T0) one can have the following sequences
  // where: T is autotrigger, A is ALICE trigger, P is a previous bunch, - is skipped
  // ---PA      A0 || A1
  // ---TA      A0 || A1
  // --TPA      A2 && T0
  // -TPPA      A2 && TM        A3 && T0
  // On the other hand, in autotrigger mode one can have a gap
  // ---PT
  // --PTT
  // -PTPT
  // PT-PT
  // therefore we have to look for an interaction outside the range of consecutive bunch
  // crossings that is used in reconstruction. Therefore the correction is done outside
  // reconstruction loop
  // In case TDC correction parameters are missing (e.g. mTDCCorr==0) then
  // pile-up is flagged but not corrected for

#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  LOG(info) << "________________________________________________________________________________";
  LOG(info) << __func__;
#endif

  // TODO: Perform actual pile-up correction for TDCs.. this is still work in progress..
  // For the moment this function has pile-up detection

  for (int itdc = 0; itdc < NTDCChannels; itdc++) {
    // Queue is empty at first event of the time frame
    // TODO: collect information from previous time frame
    std::deque<DigiRecoTDC> tdc;
    for (int ibc = 0; ibc < mNBC; ibc++) {
      // Bunch to be corrected
      auto rec = &mReco[ibc];
      // Count the number of hits in preceding bunch crossings
      // N.B. it is initialized at every bunch crossing
      int ntdc[NBCAn] = {0};
      for (auto it = tdc.begin(); it != tdc.end(); ++it) {
        auto bcd = (rec->ir).differenceInBC((*it).ir);
        if (bcd > NBCAn) {
          // Remove early events
          tdc.pop_front();
        } else if (bcd > 0) {
          ntdc[bcd - 1]++;
        }
      }
      if (rec->ntdc[itdc] > 1) {
        // In-bunch pile-up: cannot correct
        rec->tdcPileEvE[TDCSignal[itdc]] = true;
        if (mRopt->doStoreEvPileup) {
          // Store also multiple hits
          for (int ihit = 0; ihit < rec->ntdc[itdc]; ihit++) {
            tdc.emplace_back(rec->TDCVal[itdc][ihit], rec->TDCAmp[itdc][ihit], rec->ir);
          }
        }
      } else if (rec->ntdc[itdc] == 1) {
        // A single TDC hit is present in current bunch
        if (tdc.size() > 0) {
          if (mCorrBackground) {
            correctTDCBackground(ibc, itdc, tdc);
          } else {
            // Identify pile-up and assign error flags
            std::deque<DigiRecoTDC>::reverse_iterator rit = tdc.rbegin();
            // Here we should start the loop on signal bucket position
            for (rit = tdc.rbegin(); rit != tdc.rend(); ++rit) {
              auto bcd = (rec->ir).differenceInBC((*rit).ir);
              // Check if background can be corrected
              if (bcd > 0 && bcd <= NBCAn) {
                if (ntdc[bcd - 1] > 0) {
                  // We flag pile-up
                  if (bcd == 1) {
                    rec->tdcPileM1E[TDCSignal[itdc]] = true;
                  } else if (bcd == 2) {
                    rec->tdcPileM2E[TDCSignal[itdc]] = true;
                  } else if (bcd == 3) {
                    rec->tdcPileM3E[TDCSignal[itdc]] = true;
                  }
                }
              }
            }
          }
        }
        // Add current event at the end of the queue
        for (int ihit = 0; ihit < rec->ntdc[itdc]; ihit++) {
          tdc.emplace_back(rec->TDCVal[itdc][ihit], rec->TDCAmp[itdc][ihit], rec->ir);
        }
      } // Single hit in bunch
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
      if (rec->tdcPileEvE[TDCSignal[itdc]] || rec->tdcPileM1C[TDCSignal[itdc]] || rec->tdcPileM1E[TDCSignal[itdc]] ||
          rec->tdcPileM2C[TDCSignal[itdc]] || rec->tdcPileM2E[TDCSignal[itdc]] || rec->tdcPileM3C[TDCSignal[itdc]] || rec->tdcPileM3E[TDCSignal[itdc]]) {
        printf("%d %u.%-4u %s:%s%s%s%s\n", ibc, rec->ir.orbit, rec->ir.bc, ChannelNames[TDCSignal[itdc]].data(),
               rec->tdcPileEvE[TDCSignal[itdc]] ? " PileEvE" : "",
               rec->tdcPileM1C[TDCSignal[itdc]] ? " PileM1C" : "",
               rec->tdcPileM1E[TDCSignal[itdc]] ? " PileM1E" : "",
               rec->tdcPileM2C[TDCSignal[itdc]] ? " PileM2C" : "",
               rec->tdcPileM2E[TDCSignal[itdc]] ? " PileM2E" : "",
               rec->tdcPileM3C[TDCSignal[itdc]] ? " PileM3C" : "",
               rec->tdcPileM3E[TDCSignal[itdc]] ? " PileM3E" : "");
      }
#endif
    }
  }
} // correctTDCPile

// #define O2_ZDC_DEBUG_CORR

int DigiReco::correctTDCSignal(int itdc, int16_t TDCVal, float TDCAmp, float& fTDCVal, float& fTDCAmp, bool isbeg, bool isend)
{
  // Correction of single TDC signals
  // This function takes into account the position of the signal in the sequence
  // TDCVal is before recentering
  constexpr int TDCRange = TSN * NTimeBinsPerBC;
  constexpr int TDCMax = TDCRange - TSNH - 1;

  // Fallback is no correction appliead
  fTDCVal = TDCVal;
  fTDCAmp = TDCAmp;

  if (mTDCCorr == nullptr) {
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
    printf("%21s itdc=%d TDC=%d AMP=%d MISSING mTDCCorr\n", __func__, itdc, TDCVal, TDCAmp);
#endif
    return 1;
  }

  if (isbeg == false && isend == false) {
    // Mid bunch
    fTDCAmp = TDCAmp / mTDCCorr->mAFMidC[itdc][0];
    fTDCVal = TDCVal - mTDCCorr->mTSMidC[itdc][0];
  } else if (isbeg == true) {
    {
      auto p0 = mTDCCorr->mTSBegC[itdc][0];
      auto p1 = mTDCCorr->mTSBegC[itdc][1];
      if (TDCVal > TSNH && TDCVal < p0) {
        auto diff = TDCVal - p0;
        auto p2 = mTDCCorr->mTSBegC[itdc][2];
        auto p3 = mTDCCorr->mTSBegC[itdc][3];
        fTDCVal = TDCVal - (p1 + p2 * diff + p3 * diff * diff);
      } else {
        fTDCVal = TDCVal - p1;
      }
    }
    {
      auto p0 = mTDCCorr->mAFBegC[itdc][0];
      auto p1 = mTDCCorr->mAFBegC[itdc][1];
      if (TDCVal > TSNH && TDCVal < p0) {
        auto diff = TDCVal - p0;
        auto p2 = mTDCCorr->mAFBegC[itdc][2];
        auto p3 = mTDCCorr->mAFBegC[itdc][3];
        fTDCAmp = TDCAmp / (p1 + p2 * diff + p3 * diff * diff);
      } else {
        fTDCAmp = TDCAmp / p1;
      }
    }
  } else if (isend == true) {
    {
      auto p0 = mTDCCorr->mTSEndC[itdc][0];
      auto p1 = mTDCCorr->mTSEndC[itdc][1];
      if (TDCVal > p0 && TDCVal < TDCMax) {
        auto diff = TDCVal - p0;
        auto p2 = mTDCCorr->mTSEndC[itdc][2];
        auto p3 = mTDCCorr->mTSEndC[itdc][3];
        fTDCVal = TDCVal - (p1 + p2 * diff + p3 * diff * diff);
      } else {
        fTDCVal = TDCVal - p1;
      }
    }
    {
      auto p0 = mTDCCorr->mAFEndC[itdc][0];
      auto p1 = mTDCCorr->mAFEndC[itdc][1];
      if (TDCVal > p0 && TDCVal < TDCMax) {
        auto diff = TDCVal - p0;
        auto p2 = mTDCCorr->mAFEndC[itdc][2];
        auto p3 = mTDCCorr->mAFEndC[itdc][3];
        fTDCAmp = TDCAmp / (p1 + p2 * diff + p3 * diff * diff);
      } else {
        fTDCAmp = TDCAmp / p1;
      }
    }
  } else {
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
    printf("%21s itdc=%d TDC=%d AMP=%d LONELY BUNCH\n", __func__, itdc, TDCVal, TDCAmp);
#endif
    return 1;
  }
  return 0;
} // correctTDCSignal

int DigiReco::correctTDCBackground(int ibc, int itdc, std::deque<DigiRecoTDC>& tdc)
{
  constexpr int TDCRange = TSN * NTimeBinsPerBC;
  constexpr int BucketS = TDCRange / NBucket;
  constexpr int BucketSH = BucketS / 2;
  auto rec = &mReco[ibc];
  auto isig = TDCSignal[itdc];
  // With this function we are able to correct a single hit in a bunch
  // therefore we refer just to TDC hit in position [0]
  float TDCValUnc = rec->TDCVal[itdc][0];
  float TDCAmpUnc = rec->TDCAmp[itdc][0];
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  auto TDCValUncBck = rec->TDCVal[itdc][0];
  auto TDCAmpUncBck = rec->TDCAmp[itdc][0];
#endif
  float TDCValBest = TDCValUnc;
  float TDCAmpBest = TDCAmpUnc;
  int TDCSigBest = -1;
  float dtime = std::numeric_limits<float>::infinity();
  // Try every bucket position
  for (int ibuks = 0; ibuks < NBucket; ibuks++) {
    std::deque<DigiRecoTDC>::reverse_iterator rit = tdc.rbegin();
    // Correct amplitude
    float sum[3] = {0};
    // Loop on background signals to sum the effects on reconstructed amplitude
    int32_t nbkg = 0;
    for (rit = tdc.rbegin(); rit != tdc.rend(); ++rit) {
      auto bcd = (rec->ir).differenceInBC((*rit).ir);
      if (bcd > 0 && bcd <= NBCAn) {
        // Flag error if correction object does not exists
        if (mTDCCorr == nullptr) {
          if (bcd == 1) {
            rec->tdcPileM1E[isig] = true;
          } else if (bcd == 2) {
            rec->tdcPileM2E[isig] = true;
          } else if (bcd == 3) {
            rec->tdcPileM3E[isig] = true;
          }
        } else {
          int16_t TDCBkgVal = (*rit).val;
          int16_t TDCBkgAmp = (*rit).amp;
          // Get bucket number for background with floor division (divisor is positive)
          int arg1 = TDCBkgVal + BucketSH;
          int q1 = arg1 / BucketS;
          int r1 = arg1 % BucketS;
          if (r1 < 0) {
            q1--;
            r1 += TSN;
          }
          int arg2 = q1 + NBKZero;
          int q2 = arg2 / NBucket;
          int ibukb = arg2 % NBucket;
          if (ibukb < 0) {
            q2--;
            ibukb += NBucket;
          }
          // Correction parameters for amplitude
          int32_t ibun = NBCAn - bcd;
          auto p0 = mTDCCorr->mAmpCorr[itdc][ibun][ibukb][ibuks][0];
          auto p1 = mTDCCorr->mAmpCorr[itdc][ibun][ibukb][ibuks][1];
          auto p2 = mTDCCorr->mAmpCorr[itdc][ibun][ibukb][ibuks][2];
#ifdef O2_ZDC_DEBUG_CORR
          printf("%+e,%+e,%+e, // as%d_bc%+d_bk%d_sn%d\n", p0, p1, p2, itdc, -bcd, ibukb, ibuks);
#endif
          // Flag error if parameters are NaN
          if (std::isnan(p0) || std::isnan(p1) || std::isnan(p2)) {
            if (bcd == 1) {
              rec->tdcPileM1E[isig] = true;
            } else if (bcd == 2) {
              rec->tdcPileM2E[isig] = true;
            } else if (bcd == 3) {
              rec->tdcPileM3E[isig] = true;
            }
          } else {
            nbkg++;
            sum[0] += p0;
            sum[1] += p1;
            sum[2] += p2 * TDCBkgAmp;
            // Flag application of correction and flag error if
            // there are multiple signals in a bunch
            if (ibuks == 0) { // Make the test just once
              if (bcd == 1) {
                if (rec->tdcPileM1C[isig]) {
                  rec->tdcPileM1E[isig] = true;
                } else {
                  rec->tdcPileM1C[isig] = true;
                }
              } else if (bcd == 2) {
                if (rec->tdcPileM2C[isig]) {
                  rec->tdcPileM2E[isig] = true;
                } else {
                  rec->tdcPileM2C[isig] = true;
                }
              } else if (bcd == 3) {
                if (rec->tdcPileM3C[isig]) {
                  rec->tdcPileM3E[isig] = true;
                } else {
                  rec->tdcPileM3C[isig] = true;
                }
              }
            }
          }
        }
      }
    }
    if (mTDCCorr == nullptr) {
      // Cannot correct and error conditions has been already set above
      return 1;
    }
    if (nbkg > 0) { // Cross check.. should always be true
      float TDCAmpUpd = (TDCAmpUnc - sum[0] - sum[2]) / (1. - float(nbkg) + sum[1]);
      // Compute time correction assuming that time shift is additive
      float tshift = 0;
      for (rit = tdc.rbegin(); rit != tdc.rend(); ++rit) {
        auto bcd = (rec->ir).differenceInBC((*rit).ir);
        if (bcd > 0 && bcd <= NBCAn) {
          int16_t TDCBkgVal = (*rit).val;
          int16_t TDCBkgAmp = (*rit).amp;
          // Get bucket number for background with floor division (divisor is positive)
          int arg1 = TDCBkgVal + BucketSH;
          int q1 = arg1 / BucketS;
          int r1 = arg1 % BucketS;
          if (r1 < 0) {
            q1--;
            r1 += TSN;
          }
          int arg2 = q1 + NBKZero;
          int q2 = arg2 / NBucket;
          int ibukb = arg2 % NBucket;
          if (ibukb < 0) {
            q2--;
            ibukb += NBucket;
          }
          // Parameters for time correction
          int32_t ibun = NBCAn - bcd;
          auto p0 = mTDCCorr->mTDCCorr[itdc][ibun][ibukb][ibuks][0];
          auto p1 = mTDCCorr->mTDCCorr[itdc][ibun][ibukb][ibuks][1];
          auto p2 = mTDCCorr->mTDCCorr[itdc][ibun][ibukb][ibuks][2];
#ifdef O2_ZDC_DEBUG_CORR
          printf("%+e,%+e,%+e, // ts%d_bc%d_bk%d_sn%d\n", p0, p1, p2, itdc, -bcd, ibukb, ibuks);
#endif
          // Flag error if parameters are NaN
          if (std::isnan(p0) || std::isnan(p1)) {
            if (bcd == 1) {
              rec->tdcPileM1E[isig] = true;
            } else if (bcd == 2) {
              rec->tdcPileM2E[isig] = true;
            } else if (bcd == 3) {
              rec->tdcPileM3E[isig] = true;
            }
          } else {
            tshift += p0 + p1 / (TDCAmpUpd / TDCBkgAmp);
#ifdef O2_ZDC_DEBUG_CORR
            printf("ibuk = b.%d s.%d = %8.2f ts=%8.2f\n", ibukb, ibuks, TDCAmpUpd, TDCBkgAmp, tshift);
#endif
          }
        }
      }
      // Update signal arrival time
      float TDCValUpd = TDCValUnc - tshift;
      float TDCBucket = (ibuks - NBKZero) * BucketS;
      // Compare updated arrival time with assumed bucket assignment
      // Take into account the possibility that the TDC has been assigned to preceding
      // or successive bunch
      float mydtime = std::min(std::abs(TDCBucket - TDCValUpd), std::min(std::abs(TDCBucket - TDCValUpd - TDCRange), std::abs(TDCBucket - TDCValUpd + TDCRange)));
#ifdef O2_ZDC_DEBUG_CORR
      printf("ibuks = %d = %8.2f AS=%8.2f ts=%8.2f TDC %8.2f -> %8.2f delta=%8.2f\n", ibuks, TDCBucket, TDCAmpUpd, tshift, TDCValUnc, TDCValUpd, mydtime);
#endif
      if (mydtime < dtime) {
        dtime = mydtime;
        TDCValBest = TDCValUpd;
        TDCAmpBest = TDCAmpUpd;
        TDCSigBest = ibuks;
      }
    }
  } // Loop on signal bucket position (ibuks)
  rec->TDCVal[itdc][0] = TDCValBest;
  rec->TDCAmp[itdc][0] = TDCAmpBest;
#ifdef ALICEO2_ZDC_DIGI_RECO_DEBUG
  if (rec->TDCVal[itdc][0] != TDCValUnc || rec->TDCAmp[itdc][0] != TDCAmpUnc) {
    printf("%21s ibc=%d itdc=%d sn = %d", __func__, ibc, itdc, TDCSigBest);
    printf(" TDC=%f -> %f", TDCValUncBck, rec->TDCVal[itdc][0]);
    printf(" AMP=%f -> %f\n", TDCAmpUncBck, rec->TDCAmp[itdc][0]);
  }
#endif
  return 0;
}
} // namespace zdc
} // namespace o2
