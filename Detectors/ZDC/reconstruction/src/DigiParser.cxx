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
#include "ZDCReconstruction/DigiParser.h"
#include "ZDCReconstruction/RecoParamZDC.h"

namespace o2
{
namespace zdc
{

void DigiReco::init()
{
  LOG(info) << "Initialization of ZDC DigiParser";
  if (!mModuleConfig) {
    LOG(fatal) << "Missing ModuleConfig configuration object";
    return;
  }

  mTriggerMask = mModuleConfig->getTriggerMask();
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
            rec.ezdc[ich] = (sum - mRopt->adc_offset[ich]) * mRopt->energy_calib[ich];
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

} // namespace zdc
} // namespace o2
