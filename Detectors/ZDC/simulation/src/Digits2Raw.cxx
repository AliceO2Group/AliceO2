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

#include <string>
#include <TFile.h>
#include <TTree.h>
#include <TRandom.h>
#include <TMath.h>
#include "ZDCBase/Constants.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCSimulation/Digitizer.h"
#include "ZDCSimulation/Digits2Raw.h"
#include "ZDCSimulation/ZDCSimParam.h"
#include "CommonUtils/StringUtils.h"
#include "Framework/Logger.h"

using namespace o2::zdc;

//______________________________________________________________________________
void Digits2Raw::processDigits(const std::string& outDir, const std::string& fileDigitsName)
{
  auto& sopt = ZDCSimParam::Instance();
  mIsContinuous = sopt.continuous;

  if (!mModuleConfig) {
    LOG(fatal) << "Missing ModuleConfig configuration object";
    return;
  }

  if (!mSimCondition) {
    LOG(fatal) << "Missing SimCondition configuration object";
    return;
  }

  if (mNEmpty < 0) {
    LOG(fatal) << "Bunch crossing map is not initialized";
    return;
  }

  if (mNEmpty == 0) {
    LOG(warning) << "Bunch crossing map has zero clean empty bunches";
  }

  setTriggerMask();

  std::string outd = outDir;
  if (outd.back() != '/') {
    outd += '/';
  }

  mLinkID = uint32_t(0);
  mCruID = uint16_t(0);
  mFLPID = uint16_t(0);
  mEndPointID = uint32_t(0);
  // TODO: assign FeeID from configuration object
  // N.B. Now the electronics has the possibility to reconfigure FEE ID in order to match
  // what is expected from simulation. The FEE ID should never change in the future
  for (int ilink = 0; ilink < NLinks; ilink++) {
    uint64_t FeeID = uint64_t(ilink);
    std::string outFileLink = o2::utils::Str::concat_string(outDir, "/", "ZDC");
    if (mFileFor != "all") {
      outFileLink += fmt::format("_{}", mFLP);
      if (mFileFor != "flp") {
        outFileLink += fmt::format("_cru{}_{}", mCruID, mEndPointID);
        if (mFileFor != "cruendpoint") {
          outFileLink += fmt::format("_lnk{}_feeid{}", ilink, FeeID);
          if (mFileFor != "link") {
            LOG(fatal) << "Not supported output file splitting: " << mFileFor;
            throw std::runtime_error("invalid option provided for file grouping");
          }
        }
      }
    }
    outFileLink += ".raw";
    mWriter.registerLink(FeeID, mCruID, mLinkID, mEndPointID, outFileLink);
  }

  std::unique_ptr<TFile> digiFile(TFile::Open(fileDigitsName.c_str()));
  if (!digiFile || digiFile->IsZombie()) {
    LOG(fatal) << "Failed to open input digits file " << fileDigitsName;
    return;
  }

  TTree* digiTree = (TTree*)digiFile->Get("o2sim");
  if (!digiTree) {
    LOG(fatal) << "Failed to get digits tree";
    return;
  }

  if (digiTree->GetBranch("ZDCDigitBC")) {
    digiTree->SetBranchAddress("ZDCDigitBC", &mzdcBCDataPtr);
  } else {
    LOG(fatal) << "Branch ZDCDigitBC is missing";
    return;
  }

  if (digiTree->GetBranch("ZDCDigitCh")) {
    digiTree->SetBranchAddress("ZDCDigitCh", &mzdcChDataPtr);
  } else {
    LOG(fatal) << "Branch ZDCDigitCh is missing";
    return;
  }

  if (digiTree->GetBranch("ZDCDigitOrbit")) {
    digiTree->SetBranchAddress("ZDCDigitOrbit", &mzdcPedDataPtr);
  } else {
    LOG(fatal) << "Branch ZDCDigitOrbit is missing";
    return;
  }

  if (digiTree->GetBranchStatus("ZDCDigitLabels")) {
    digiTree->SetBranchStatus("ZDCDigitLabel*", 0);
  }
  int nBCadd = 0;
  // TODO: fix the conversion logic
  // If first stored bunch is not in first digitized orbit the empty orbits
  // at the beginning of the simulation are not converted to raw data
  // If last orbits are empty they are not inserted in raw data
  // Logic is not tested if digiTree has more than one entry
  // The resulting file has less orbits. However the missing orbits have no signals
  // and threefore are not used in reconstruction -> this fix has low priority
  for (int ient = 0; ient < digiTree->GetEntries(); ient++) {
    digiTree->GetEntry(ient);
    mNbc = mzdcBCData.size();
    if (mVerbosity > 0) {
      LOG(info) << "Entry " << ient << ": processing " << mNbc << " BCs stored";
    }
    for (int ibc = 0; ibc < mNbc; ibc++) {
      mBCD = mzdcBCData[ibc];
      convertDigits(ibc);
      writeDigits();
      // Detect last event or orbit change and insert last bunch
      if (ibc == (mNbc - 1)) {
        // For last event we need to close last orbit (if it is needed)
        if (mzdcBCData[ibc].ir.bc != 3563) {
          if (mVerbosity > 1) {
            LOG(info) << "Closing last orbit " << mzdcBCData[ibc].ir.orbit;
          }
          insertLastBunch(ibc, mzdcBCData[ibc].ir.orbit);
          writeDigits();
          nBCadd++;
        }
      } else {
        auto this_orbit = mzdcBCData[ibc].ir.orbit;
        auto next_orbit = mzdcBCData[ibc + 1].ir.orbit;
        // If current bunch is last bunch in the orbit we don't insert it again
        if (mzdcBCData[ibc].ir.bc == 3563) {
          this_orbit = this_orbit + 1;
        }
        // We may need to insert more than one orbit
        for (auto orbit = this_orbit; orbit < next_orbit; orbit++) {
          if (mVerbosity > 1) {
            LOG(info) << "Inserting last bunch for orbit " << orbit;
          }
          insertLastBunch(ibc, orbit);
          writeDigits();
          nBCadd++;
        }
      }
    }
    LOG(info) << "Entry " << ient << " Converted BCs: " << mNbc << " + added@3563:" << nBCadd << " = " << mNbc + nBCadd;
  }
  digiFile->Close();
}

//______________________________________________________________________________
void Digits2Raw::setTriggerMask()
{
  mTriggerMask = 0;
  mPrintTriggerMask = "";
  for (int32_t im = 0; im < NModules; im++) {
    if (im > 0) {
      mPrintTriggerMask += " ";
    }
    mPrintTriggerMask += std::to_string(im);
    mPrintTriggerMask += "[";
    for (uint32_t ic = 0; ic < NChPerModule; ic++) {
      if (mModuleConfig->modules[im].trigChannel[ic]) {
        uint32_t tmask = 0x1 << (im * NChPerModule + ic);
        mTriggerMask = mTriggerMask | tmask;
        mPrintTriggerMask += "T";
      } else {
        mPrintTriggerMask += " ";
      }
    }
    mPrintTriggerMask += "]";
#ifdef O2_ZDC_DEBUG
    uint32_t mytmask = mTriggerMask >> (im * NChPerModule);
    printf("Trigger mask for module %d 0123 %s%s%s%s\n", im,
           mytmask & 0x1 ? "T" : "N",
           mytmask & 0x2 ? "T" : "N",
           mytmask & 0x4 ? "T" : "N",
           mytmask & 0x8 ? "T" : "N");
#endif
  }
  printf("trigger_mask=0x%08x %s", mTriggerMask, mPrintTriggerMask.data());
}

//______________________________________________________________________________
inline void Digits2Raw::resetSums(uint32_t orbit)
{
  for (int32_t im = 0; im < NModules; im++) {
    for (int32_t ic = 0; ic < NChPerModule; ic++) {
      mScalers[im][ic] = 0;
      mSumPed[im][ic] = 0;
      mPed[im][ic] = 0;
    }
  }
  mLatestOrbit = orbit;
  mLastNEmpty = 0;
}

//______________________________________________________________________________
inline void Digits2Raw::updatePedestalReference(int bc)
{
  // Compute or update baseline reference
  // In the last BC we copy what is stored in the digits
  if (bc == 3563) {
    int io = 0;
    for (; io < mzdcPedData.size(); io++) {
      uint32_t orbit = mBCD.ir.orbit;
      if (orbit == mzdcPedData[io].ir.orbit) {
        break;
      }
    }
    if (io == mzdcPedData.size()) {
      LOG(fatal) << "Cannot find orbit";
    }
    for (int32_t im = 0; im < NModules; im++) {
      for (int32_t ic = 0; ic < NChPerModule; ic++) {
        // Identify connected channel
        auto id = mModuleConfig->modules[im].channelID[ic];
        mPed[im][ic] = *((uint16_t*)&mzdcPedData[io].data[id]);
      }
    }
  } else if (mEmpty[bc] > 0 && mEmpty[bc] != mLastNEmpty) {
    // For the preceding bunch crossing we make-up the fields in a random walk
    // fashion like in the hardware. The result however cannot be coherent with
    // what is stored in the last bunch
    for (int32_t im = 0; im < NModules; im++) {
      for (int32_t ic = 0; ic < NChPerModule; ic++) {
        // Identify connected channel
        auto id = mModuleConfig->modules[im].channelID[ic];
        auto base_m = mSimCondition->channels[id].pedestal;      // Average pedestal
        auto base_s = mSimCondition->channels[id].pedestalFluct; // Baseline oscillations
        auto base_n = mSimCondition->channels[id].pedestalNoise; // Electronic noise
        double deltan = mEmpty[bc] - mLastNEmpty;
        // We assume to have a fluctuation every two bunch crossings
        // Will need to tune this parameter
        double k = 2.;
        mSumPed[im][ic] += gRandom->Gaus(12. * deltan * base_m, 12. * k * base_s * TMath::Sqrt(deltan / k));
        // Adding in quadrature the RMS of pedestal electronic noise
        mSumPed[im][ic] += gRandom->Gaus(0, base_n * TMath::Sqrt(12. * deltan));
        double myped = mSumPed[im][ic] / double(mEmpty[bc]) / 12.;  // Average current pedestal
        myped = TMath::Nint(myped / mModuleConfig->baselineFactor); // Convert into digitized pedestal
        int16_t theped = myped;
        // Correct for overflow and underflow
        if (myped < -32768) {
          theped = -32768;
        }
        if (myped > 32767) {
          myped = 32767;
        }
        mPed[im][ic] = *((uint16_t*)&theped);
      }
    }
    mLastNEmpty = mEmpty[bc];
  }
}

//______________________________________________________________________________
inline void Digits2Raw::resetOutputStructure(uint16_t bc, uint32_t orbit, bool is_dummy)
{
  // Increment scalers and reset output structure
  for (uint32_t im = 0; im < NModules; im++) {
    for (uint32_t ic = 0; ic < NChPerModule; ic++) {
      // Fixed words
      mZDC.data[im][ic].w[0][0] = Id_w0;
      mZDC.data[im][ic].w[0][1] = 0;
      mZDC.data[im][ic].w[0][2] = 0;
      mZDC.data[im][ic].w[0][3] = 0;
      mZDC.data[im][ic].w[1][0] = Id_w1;
      mZDC.data[im][ic].w[1][1] = 0;
      mZDC.data[im][ic].w[1][2] = 0;
      mZDC.data[im][ic].w[1][3] = 0;
      mZDC.data[im][ic].w[2][0] = Id_w2;
      mZDC.data[im][ic].w[2][1] = 0;
      mZDC.data[im][ic].w[2][2] = 0;
      mZDC.data[im][ic].w[2][3] = 0;
      // Module and channel numbers
      mZDC.data[im][ic].f.board = im;
      mZDC.data[im][ic].f.ch = ic;
      // Orbit and bunch crossing
      mZDC.data[im][ic].f.orbit = orbit;
      mZDC.data[im][ic].f.bc = bc;
      // If channel is hit in current bunch crossing
      if (!is_dummy) {
        if (mBCD.triggers & (0x1 << (im * NChPerModule + ic))) {
          mScalers[im][ic]++;          // increment scalers
          mZDC.data[im][ic].f.Hit = 1; // flag bunch crossing
        }
      }
      mZDC.data[im][ic].f.hits = mScalers[im][ic];
      mZDC.data[im][ic].f.offset = mPed[im][ic];
    }
  }
}

//______________________________________________________________________________
inline void Digits2Raw::assignTriggerBits(int ibc, uint16_t bc, uint32_t orbit, bool is_dummy)
{
  // Triggers refer to the HW trigger conditions (32 possible channels)
  // Autotrigger, current bunch crossing
  ModuleTriggerMapData triggers;
  // Autotrigger and ALICE trigger bits are zero for a dummy bunch crossing
  if (!is_dummy) {
    for (uint32_t im = 0; im < NModules; im++) {
      triggers.w = mzdcBCData[ibc].moduleTriggers[im];
      for (uint32_t ic = 0; ic < NChPerModule; ic++) {
        mZDC.data[im][ic].f.Alice_0 = triggers.f.Alice_0;
        mZDC.data[im][ic].f.Alice_1 = triggers.f.Alice_1;
        mZDC.data[im][ic].f.Alice_2 = triggers.f.Alice_2;
        mZDC.data[im][ic].f.Alice_3 = triggers.f.Alice_3;
        mZDC.data[im][ic].f.Auto_m = triggers.f.Auto_m;
        mZDC.data[im][ic].f.Auto_0 = triggers.f.Auto_0;
        mZDC.data[im][ic].f.Auto_1 = triggers.f.Auto_1;
        mZDC.data[im][ic].f.Auto_2 = triggers.f.Auto_2;
        mZDC.data[im][ic].f.Auto_3 = triggers.f.Auto_3;
      }
    }
  }
}

//______________________________________________________________________________
void Digits2Raw::insertLastBunch(int ibc, uint32_t orbit)
{
  // Inserting last bunch in the orbit. This is not present in digits because information is stored in the
  // OrbitData structure

  // Orbit and bunch crossing identifiers
  uint16_t bc = 3563;

  // Reset scalers at orbit change
  if (orbit != mLatestOrbit) {
    resetSums(orbit);
  }

  updatePedestalReference(bc);

  // Dummy bunch -> Do not increment scalers but reset output structure
  resetOutputStructure(bc, orbit, true);

  // Compute autotrigger bits and assign ALICE trigger bits
  assignTriggerBits(ibc, bc, orbit, true);

  // Insert payload for all channels
  for (int32_t im = 0; im < NModules; im++) {
    for (uint32_t ic = 0; ic < NChPerModule; ic++) {
      if (mModuleConfig->modules[im].readChannel[ic]) {
        auto id = mModuleConfig->modules[im].channelID[ic];
        auto base_m = mSimCondition->channels[id].pedestal;      // Average pedestal
        auto base_s = mSimCondition->channels[id].pedestalFluct; // Baseline oscillations
        auto base_n = mSimCondition->channels[id].pedestalNoise; // Electronic noise
        double base = gRandom->Gaus(base_m, base_s);
        double val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s00 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s01 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s02 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s03 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s04 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s05 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s06 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s07 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s08 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s09 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s10 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s11 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
      }
    }
  }
} // insertLastBunch

//______________________________________________________________________________
void Digits2Raw::convertDigits(int ibc)
{
  // Creating raw data from a bunch crossing that is actually present in digits

  // Orbit and bunch crossing identifiers
  uint16_t bc = mBCD.ir.bc;
  uint32_t orbit = mBCD.ir.orbit;

  // Reset scalers at orbit change
  if (orbit != mLatestOrbit) {
    resetSums(orbit);
  }

  updatePedestalReference(bc);

  // Not a dummy bunch -> Reset output structure and eventually flag hits and increment scalers
  resetOutputStructure(bc, orbit, false);

  // Compute autotrigger bits and assign ALICE trigger bits
  assignTriggerBits(ibc, bc, orbit, false);

  if (mVerbosity > 0) {
    mBCD.print(mTriggerMask);
  }

  int chEnt = mBCD.ref.getFirstEntry();
  for (int ic = 0; ic < mBCD.ref.getEntries(); ic++) {
    const auto& chd = mzdcChData[chEnt++];
    if (mVerbosity > 0) {
      chd.print();
    }
    uint16_t bc = mBCD.ir.bc;
    uint32_t orbit = mBCD.ir.orbit;
    // Look for channel ID in digits and store channel (just one copy in output)
    // This is a limitation of software but we are not supposed to acquire the
    // same signal twice anyway
    for (int32_t im = 0; im < NModules; im++) {
      for (uint32_t ic = 0; ic < NChPerModule; ic++) {
        if (mModuleConfig->modules[im].channelID[ic] == chd.id &&
            mModuleConfig->modules[im].readChannel[ic]) {
          int32_t is = 0;
          mZDC.data[im][ic].f.s00 = chd.data[is++];
          mZDC.data[im][ic].f.s01 = chd.data[is++];
          mZDC.data[im][ic].f.s02 = chd.data[is++];
          mZDC.data[im][ic].f.s03 = chd.data[is++];
          mZDC.data[im][ic].f.s04 = chd.data[is++];
          mZDC.data[im][ic].f.s05 = chd.data[is++];
          mZDC.data[im][ic].f.s06 = chd.data[is++];
          mZDC.data[im][ic].f.s07 = chd.data[is++];
          mZDC.data[im][ic].f.s08 = chd.data[is++];
          mZDC.data[im][ic].f.s09 = chd.data[is++];
          mZDC.data[im][ic].f.s10 = chd.data[is++];
          mZDC.data[im][ic].f.s11 = chd.data[is++];
          break;
        }
      }
    }
  }
}

//______________________________________________________________________________
void Digits2Raw::writeDigits()
{
  constexpr static int data_size = sizeof(uint32_t) * NWPerGBTW;
  constexpr static gsl::span<char> empty;
  // Local interaction record (true and empty bunches)
  o2::InteractionRecord ir(mZDC.data[0][0].f.bc, mZDC.data[0][0].f.orbit);
  for (uint32_t im = 0; im < o2::zdc::NModules; im++) {
    // Check if module has been filled with data
    // N.B. All channels are initialized if module is supposed to be readout
    // Trigger bits are the same for all the channels connected to a module
    bool TM = mZDC.data[im][0].f.Auto_m;
    bool T0 = mZDC.data[im][0].f.Auto_0;
    bool T1 = mZDC.data[im][0].f.Auto_1;
    bool T2 = mZDC.data[im][0].f.Auto_2;
    bool T3 = mZDC.data[im][0].f.Auto_3;
    bool A0 = mZDC.data[im][0].f.Alice_0;
    bool A1 = mZDC.data[im][0].f.Alice_1;
    bool A2 = mZDC.data[im][0].f.Alice_2;
    bool A3 = mZDC.data[im][0].f.Alice_3;
    bool tcond_continuous = T0 || T1;
    bool tcond_triggered = A0 || A1 || (A2 && (T0 || TM)) || (A3 && T0);
    bool tcond_last = mZDC.data[im][0].f.bc == 3563;
    // Condition to write GBT data
    bool addedChData[NChPerModule] = {false, false, false, false};
    if (tcond_triggered || (mIsContinuous && tcond_continuous) || (mZDC.data[im][0].f.bc == 3563)) {
      for (uint32_t ic = 0; ic < o2::zdc::NChPerModule; ic++) {
        uint64_t FeeID = 2 * im + ic / 2;
        if (mModuleConfig->modules[im].readChannel[ic]) {
          for (int32_t iw = 0; iw < o2::zdc::NWPerBc; iw++) {
            if (mEnablePadding) {
              gsl::span<char> payload{reinterpret_cast<char*>(&mZDC.data[im][ic].w[iw][0]), data_size};
              mWriter.addData(FeeID, mCruID, mLinkID, mEndPointID, ir, payload);
            } else {
              gsl::span<char> payload{reinterpret_cast<char*>(&mZDC.data[im][ic].w[iw][0]), PayloadPerGBTW};
              o2::zdc::Digits2Raw::print_gbt_word((const uint32_t*)&mZDC.data[im][ic].w[iw][0]);
              mWriter.addData(FeeID, mCruID, mLinkID, mEndPointID, ir, payload);
            }
          }
          addedChData[ic] = true;
        }
      }
    }
    // All links are registered, we add explicitly zero payload
    if (addedChData[0] == false && addedChData[1] == false) {
      uint64_t FeeID = 2 * im;
      mWriter.addData(FeeID, mCruID, mLinkID, mEndPointID, ir, empty);
    }
    if (addedChData[2] == false && addedChData[3] == false) {
      uint64_t FeeID = 2 * im + 1;
      mWriter.addData(FeeID, mCruID, mLinkID, mEndPointID, ir, empty);
    }
    if (mVerbosity > 1) {
      if (tcond_continuous) {
        printf("M%d Cont.    T0=%d || T1=%d\n", im, T0, T1);
      }
      if (tcond_triggered) {
        printf("M%d Trig. %s A0=%d || A1=%d || (A2=%d && (T0=%d || TM=%d))=%d || (A3=%d && T0=%d )=%d\n", im, mIsContinuous ? "CM" : "TM", A0, A1, A2, T0, TM, A2 && (T0 || TM), A3, T0, A3 && T0);
      }
      if (mZDC.data[im][0].f.bc == 3563) {
        printf("M%d is last BC\n", im);
      }
      if (tcond_triggered || (mIsContinuous && tcond_continuous) || (mZDC.data[im][0].f.bc == 3563)) {
        for (uint32_t ic = 0; ic < o2::zdc::NChPerModule; ic++) {
          if (mModuleConfig->modules[im].readChannel[ic]) {
            for (int32_t iw = 0; iw < o2::zdc::NWPerBc; iw++) {
              print_gbt_word(&mZDC.data[im][ic].w[iw][0], mModuleConfig);
            }
          }
        }
      } else {
        if (mVerbosity > 2) {
          printf("orbit %9u bc %4u M%d SKIP\n", mZDC.data[im][0].f.orbit, mZDC.data[im][0].f.bc, im);
        }
      }
    }
  }
}

//______________________________________________________________________________
void Digits2Raw::print_gbt_word(const uint32_t* word, const ModuleConfig* moduleConfig)
{
  if (word == nullptr) {
    printf("NULL\n");
    return;
  }
  union {
    uint16_t uns;
    int16_t sig;
  } word16;
  unsigned __int128 val = word[2];
  val = val << 32;
  val = val | word[1];
  val = val << 32;
  val = val | word[0];
  static uint32_t last_orbit = 0, last_bc = 0;

  ULong64_t lsb = val;
  ULong64_t msb = val >> 64;
  uint32_t a = word[0];
  uint32_t b = word[1];
  uint16_t c = *((uint16_t*)&word[2]);
  // printf("\nGBTW: %04x %08x %08x\n",c,b,a);
  if ((a & 0x3) == 0) {
    uint32_t myorbit = (val >> 48) & 0xffffffff;
    uint32_t mybc = (val >> 36) & 0xfff;
    if (myorbit != last_orbit || mybc != last_bc) {
      printf("Orbit %9u bc %4u\n", myorbit, mybc);
      last_orbit = myorbit;
      last_bc = mybc;
    }
    printf("%04x %08x %08x ", c, b, a);
    uint32_t hits = (val >> 24) & 0xfff;
    word16.uns = (lsb >> 8) & 0xffff;
    // float foffset = word16.sig * (moduleConfig == nullptr ? 1 : moduleConfig->baselineFactor);
    uint32_t board = (lsb >> 2) & 0xf;
    uint32_t ch = (lsb >> 6) & 0x3;
    printf("orbit %9u bc %4u hits %4u offset %+6i Board %2u Ch %1u", myorbit, mybc, hits, word16.sig, board, ch);
    // printf("orbit %9u bc %4u hits %4u offset %+9.3f Board %2u Ch %1u", myorbit, mybc, hits, foffset, board, ch);
    if (board >= NModules) {
      printf(" ERROR with board");
    }
    if (ch >= NChPerModule) {
      printf(" ERROR with ch");
    }
    if (moduleConfig) {
      auto id = moduleConfig->modules[board].channelID[ch];
      if (id >= 0 && id < NChannels) {
        printf(" %s", ChannelNames[id].data());
      } else {
        printf(" error with ch id");
      }
    }
  } else if ((a & 0x3) == 1) {
    printf("%04x %08x %08x ", c, b, a);
    printf("     %s %s %s %s ", a & 0x10 ? "A0" : "  ", a & 0x20 ? "A1" : "  ", a & 0x40 ? "A2" : "  ", a & 0x80 ? "A3" : "  ");
    printf("0-5 ");
    int16_t s[6];
    val = val >> 8;
    for (int32_t i = 0; i < 6; i++) {
      s[i] = val & 0xfff;
      if (s[i] > ADCMax) {
        s[i] = s[i] - ADCRange;
      }
      val = val >> 12;
    }
    printf(" %5d %5d %5d %5d %5d %5d%s%s", s[0], s[1], s[2], s[3], s[4], s[5], (a & 0x4) ? " DLOSS" : "", (a & 0x8) ? " ERROR" : "");
  } else if ((a & 0x3) == 2) {
    printf("%04x %08x %08x ", c, b, a);
    printf("%s %s %s %s %s %s ", a & 0x4 ? "H" : " ", a & 0x8 ? "TM" : "  ", a & 0x10 ? "T0" : "  ", a & 0x20 ? "T1" : "  ", a & 0x40 ? "T2" : "  ", a & 0x80 ? "T3" : "  ");
    printf("6-b ");
    int16_t s[6];
    val = val >> 8;
    for (int32_t i = 0; i < 6; i++) {
      s[i] = val & 0xfff;
      if (s[i] > ADCMax) {
        s[i] = s[i] - ADCRange;
      }
      val = val >> 12;
    }
    printf(" %5d %5d %5d %5d %5d %5d", s[0], s[1], s[2], s[3], s[4], s[5]);
  } else if ((a & 0x3) == 3) {
    printf("%04x %08x %08x ", c, b, a);
    printf("HB ");
  }
  printf("\n");
}

//______________________________________________________________________________
void Digits2Raw::emptyBunches(std::bitset<3564>& bunchPattern)
{
  // Check if mModuleConfig object has list of empty bunches
  if (mModuleConfig->nBunchAverage <= 0) {
    LOG(fatal) << "nBunchAverage = " << mModuleConfig->nBunchAverage;
  }

  // Analyze the list of empty bunches that is provided
  mNEmpty = 0;
  int ib = 0;
  uint64_t one = 0x1;
  for (int i = 0; i < mModuleConfig->NWMap; i++) {
    uint64_t val = mModuleConfig->emptyMap[i];
    for (int j = 0; j < 64; j++) {
      if ((val & (one << j)) != 0) { // Empty bunch
        mNEmpty++;
        mEmpty[ib] = mNEmpty;
      } else { // Not empty
        mEmpty[ib] = mNEmpty;
      }
      ib++;
    }
  }

  // No list is provided: we prepare a list from collision context
  if (mNEmpty == 0) {
    const int LHCMaxBunches = o2::constants::lhc::LHCMaxBunches;
    for (int32_t ib = 0; ib < LHCMaxBunches; ib++) {
      int mb = (ib + 31) % o2::constants::lhc::LHCMaxBunches;                                    // beam gas from back of calorimeter (31 b.c. before)
      int m1 = (ib + 1) % o2::constants::lhc::LHCMaxBunches;                                     // previous bunch (next is colliding)
      int cb = ib;                                                                               // current bunch crossing
      int p1 = (ib - 1 + o2::constants::lhc::LHCMaxBunches) % o2::constants::lhc::LHCMaxBunches; // colliding + 1 (-1 is colliding)
      int p2 = (ib - 2 + o2::constants::lhc::LHCMaxBunches) % o2::constants::lhc::LHCMaxBunches; // colliding + 2 (-2 is colliding)
      int p3 = (ib - 3 + o2::constants::lhc::LHCMaxBunches) % o2::constants::lhc::LHCMaxBunches; // colliding + 3 (-3 is colliding)
      int p4 = (ib - 4 + o2::constants::lhc::LHCMaxBunches) % o2::constants::lhc::LHCMaxBunches; // colliding + 4 (-4 is colliding)
      if (bunchPattern[mb] || bunchPattern[m1] || bunchPattern[ib] || bunchPattern[p1] || bunchPattern[p2] || bunchPattern[p3]) {
        mEmpty[ib] = mNEmpty;
      } else {
        mNEmpty++;
        mEmpty[ib] = mNEmpty;
      }
      if (mNEmpty == mModuleConfig->nBunchAverage) {
        break;
      }
    }
  }

  // Check if there is a mismatch between requested and actual list
  if (mNEmpty != 0 && mNEmpty != mModuleConfig->nBunchAverage) {
    LOG(fatal) << "Mismatch with empty map mNEmpty = " << mNEmpty << " nBunchAverage = " << mModuleConfig->nBunchAverage;
  }
  LOG(info) << "There are " << mNEmpty << " clean empty bunches";
}
