// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "ZDCSimulation/MCLabel.h"
#include "ZDCSimulation/ZDCSimParam.h"
#include "CommonUtils/StringUtils.h"
#include "FairLogger.h"

using namespace o2::zdc;

//ClassImp(Digits2Raw);
//______________________________________________________________________________
void Digits2Raw::processDigits(const std::string& outDir, const std::string& fileDigitsName)
{
  auto& sopt = ZDCSimParam::Instance();
  mIsContinuous = sopt.continuous;

  if (!mModuleConfig) {
    LOG(FATAL) << "Missing ModuleConfig configuration object";
    return;
  }

  if (!mSimCondition) {
    LOG(FATAL) << "Missing SimCondition configuration object";
    return;
  }

  if (mNEmpty < 0) {
    LOG(FATAL) << "Bunch crossing map is not initialized";
    return;
  }

  if (mNEmpty == 0) {
    LOG(WARNING) << "Bunch crossing map has zero clean empty bunches";
  }

  setTriggerMask();

  std::string outd = outDir;
  if (outd.back() != '/') {
    outd += '/';
  }

  mLinkID = uint32_t(0);
  mCruID = uint16_t(0);
  mEndPointID = uint32_t(0);
  for (int ilink = 0; ilink < NLinks; ilink++) {
    mFeeID = uint64_t(ilink);
    std::string outFileLink = mOutputPerLink ? o2::utils::concat_string(outd, "zdc_link", std::to_string(ilink), ".raw") : o2::utils::concat_string(outd, "zdc.raw");
    mWriter.registerLink(mFeeID, mCruID, mLinkID, mEndPointID, outFileLink);
  }

  std::unique_ptr<TFile> digiFile(TFile::Open(fileDigitsName.c_str()));
  if (!digiFile || digiFile->IsZombie()) {
    LOG(ERROR) << "Failed to open input digits file " << fileDigitsName;
    return;
  }

  TTree* digiTree = (TTree*)digiFile->Get("o2sim");
  if (!digiTree) {
    LOG(ERROR) << "Failed to get digits tree";
    return;
  }

  if (digiTree->GetBranch("ZDCDigitBC")) {
    digiTree->SetBranchAddress("ZDCDigitBC", &mzdcBCDataPtr);
  } else {
    LOG(ERROR) << "Branch ZDCDigitBC is missing";
    return;
  }

  if (digiTree->GetBranch("ZDCDigitCh")) {
    digiTree->SetBranchAddress("ZDCDigitCh", &mzdcChDataPtr);
  } else {
    LOG(ERROR) << "Branch ZDCDigitCh is missing";
    return;
  }

  digiTree->SetBranchStatus("ZDCDigitLabel*", 0);

  for (int ient = 0; ient < digiTree->GetEntries(); ient++) {
    digiTree->GetEntry(ient);
    mNbc = mzdcBCData.size();
    LOG(INFO) << "Entry " << ient << " : " << mNbc << " BCs stored";
    for (int ibc = 0; ibc < mNbc; ibc++) {
      mBCD = mzdcBCData[ibc];
      convertDigits(ibc);
      writeDigits();
      // Detect last event or orbit change and insert last bunch
      if (ibc == (mNbc - 1)) {
        // For last event we need to close last orbit (if it is needed)
        if (mzdcBCData[ibc].ir.bc != 3563) {
          insertLastBunch(ibc, mzdcBCData[ibc].ir.orbit);
          writeDigits();
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
          insertLastBunch(ibc, orbit);
          writeDigits();
        }
      }
    }
  }
  digiFile->Close();
}

//______________________________________________________________________________
void Digits2Raw::setTriggerMask()
{
  mTriggerMask = 0;
  mPrintTriggerMask = "";
  for (Int_t im = 0; im < NModules; im++) {
    if (im > 0) {
      mPrintTriggerMask += " ";
    }
    mPrintTriggerMask += std::to_string(im);
    mPrintTriggerMask += "[";
    for (UInt_t ic = 0; ic < NChPerModule; ic++) {
      if (mModuleConfig->modules[im].trigChannel[ic]) {
        UInt_t tmask = 0x1 << (im * NChPerModule + ic);
        mTriggerMask = mTriggerMask | tmask;
        mPrintTriggerMask += "T";
      } else {
        mPrintTriggerMask += " ";
      }
    }
    mPrintTriggerMask += "]";
    UInt_t mytmask = mTriggerMask >> (im * NChPerModule);
    printf("Trigger mask for module %d 0123 %s%s%s%s\n", im,
           mytmask & 0x1 ? "T" : "N",
           mytmask & 0x2 ? "T" : "N",
           mytmask & 0x4 ? "T" : "N",
           mytmask & 0x8 ? "T" : "N");
  }
  printf("trigger_mask=0x%08x %s\n", mTriggerMask, mPrintTriggerMask.data());
}

//______________________________________________________________________________
inline void Digits2Raw::resetSums(uint32_t orbit)
{
  for (Int_t im = 0; im < NModules; im++) {
    for (Int_t ic = 0; ic < NChPerModule; ic++) {
      mScalers[im][ic] = 0;
      mSumPed[im][ic] = 0;
      mPed[im][ic] = 0;
    }
  }
  mLastOrbit = orbit;
  mLastNEmpty = 0;
}

//______________________________________________________________________________
inline void Digits2Raw::updatePedestalReference(int bc)
{
  // Compute or update baseline reference
  if (mEmpty[bc] > 0 && mEmpty[bc] != mLastNEmpty) {
    for (Int_t im = 0; im < NModules; im++) {
      for (Int_t ic = 0; ic < NChPerModule; ic++) {
        // Identify connected channel
        auto id = mModuleConfig->modules[im].channelID[ic];
        auto base_m = mSimCondition->channels[id].pedestal;      // Average pedestal
        auto base_s = mSimCondition->channels[id].pedestalFluct; // Baseline oscillations
        auto base_n = mSimCondition->channels[id].pedestalNoise; // Electronic noise
        Double_t deltan = mEmpty[bc] - mLastNEmpty;
        // We assume to have a fluctuation every two bunch crossings
        // Will need to tune this parameter
        Double_t k = 2.;
        mSumPed[im][ic] += gRandom->Gaus(12. * deltan * base_m, 12. * k * base_s * TMath::Sqrt(deltan / k));
        // Adding in quadrature the RMS of pedestal electronic noise
        mSumPed[im][ic] += gRandom->Gaus(0, base_n * TMath::Sqrt(12. * deltan));
        Double_t myped = TMath::Nint(8. * mSumPed[im][ic] / Double_t(mEmpty[bc]) / 12. + 32768);
        if (myped < 0) {
          myped = 0;
        }
        if (myped > 65535) {
          myped = 65535;
        }
        mPed[im][ic] = myped;
      }
    }
    mLastNEmpty = mEmpty[bc];
  }
}

//______________________________________________________________________________
inline void Digits2Raw::resetOutputStructure(UShort_t bc, UInt_t orbit, bool is_dummy)
{
  // Increment scalers and reset output structure
  for (UInt_t im = 0; im < NModules; im++) {
    for (UInt_t ic = 0; ic < NChPerModule; ic++) {
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
inline void Digits2Raw::assignTriggerBits(int ibc, UShort_t bc, UInt_t orbit, bool is_dummy)
{
  // Triggers refer to the HW trigger conditions (32 possible channels)
  // Autotrigger, current bunch crossing
  UInt_t triggers_0 = 0;
  // Autotrigger and ALICE trigger bits are zero for a dummy bunch crossing
  if (!is_dummy) {
    triggers_0 = mBCD.triggers;
    // ALICE current bunch crossing
    if (mBCD.ext_triggers) {
      for (UInt_t im = 0; im < NModules; im++) {
        for (UInt_t ic = 0; ic < NChPerModule; ic++) {
          mZDC.data[im][ic].f.Alice_0 = 1;
        }
      }
    }
  }

  // Next bunch crossings (ALICE and autotrigger)
  UInt_t triggers_1 = 0, triggers_2 = 0, triggers_3 = 0, triggers_m = 0;
  for (Int_t is = 1; is < 4; is++) {
    Int_t ibc_peek = ibc + is;
    if (ibc_peek >= mNbc) {
      break;
    }
    const auto& bcd_peek = mzdcBCData[ibc_peek];
    UShort_t bc_peek = bcd_peek.ir.bc;
    UInt_t orbit_peek = bcd_peek.ir.orbit;
    if (bcd_peek.triggers) {
      if (orbit_peek == orbit) {
        if ((bc_peek - bc) == 1) {
          triggers_1 = bcd_peek.triggers;
          if (bcd_peek.ext_triggers) {
            for (UInt_t im = 0; im < NModules; im++) {
              for (UInt_t ic = 0; ic < NChPerModule; ic++) {
                mZDC.data[im][ic].f.Alice_1 = 1;
              }
            }
          }
        } else if ((bc_peek - bc) == 2) {
          triggers_2 = bcd_peek.triggers;
          if (bcd_peek.ext_triggers) {
            for (UInt_t im = 0; im < NModules; im++) {
              for (UInt_t ic = 0; ic < NChPerModule; ic++) {
                mZDC.data[im][ic].f.Alice_2 = 1;
              }
            }
          }
        } else if ((bc_peek - bc) == 3) {
          triggers_3 = bcd_peek.triggers;
          if (bcd_peek.ext_triggers) {
            for (UInt_t im = 0; im < NModules; im++) {
              for (UInt_t ic = 0; ic < NChPerModule; ic++) {
                mZDC.data[im][ic].f.Alice_3 = 1;
              }
            }
          }
          break;
        }
      } else if (orbit_peek == (orbit + 1)) {
        if ((bc_peek + 3564 - bc) == 1) {
          triggers_1 = bcd_peek.triggers;
          if (bcd_peek.ext_triggers) {
            for (UInt_t im = 0; im < NModules; im++) {
              for (UInt_t ic = 0; ic < NChPerModule; ic++) {
                mZDC.data[im][ic].f.Alice_1 = 1;
              }
            }
          }
        } else if ((bc_peek + 3564 - bc) == 2) {
          triggers_2 = bcd_peek.triggers;
          if (bcd_peek.ext_triggers) {
            for (UInt_t im = 0; im < NModules; im++) {
              for (UInt_t ic = 0; ic < NChPerModule; ic++) {
                mZDC.data[im][ic].f.Alice_2 = 1;
              }
            }
          }
        } else if ((bc_peek + 3564 - bc) == 3) {
          triggers_3 = bcd_peek.triggers;
          if (bcd_peek.ext_triggers) {
            for (UInt_t im = 0; im < NModules; im++) {
              for (UInt_t ic = 0; ic < NChPerModule; ic++) {
                mZDC.data[im][ic].f.Alice_3 = 1;
              }
            }
          }
          break;
        }
      } else {
        break;
      }
    }
  }

  // Previous bunch crossing just for autotrigger
  // For a dummy last bunch crossing previous bunch is the one pointed by ibc
  {
    Int_t ibc_peek = is_dummy ? ibc : ibc - 1;
    if (ibc_peek >= 0) {
      const auto& bcd_peek = mzdcBCData[ibc - 1];
      UShort_t bc_peek = bcd_peek.ir.bc;
      UInt_t orbit_peek = bcd_peek.ir.orbit;
      if (bcd_peek.triggers) {
        if (orbit_peek == orbit) {
          if ((bc - bc_peek) == 1) {
            triggers_m = bcd_peek.triggers;
          }
        } else if (orbit_peek == (orbit - 1)) {
          if (bc == 0 && bc_peek == 3563) {
            triggers_m = bcd_peek.triggers;
          }
        }
      }
    }
  }

  // Assign trigger bits in payload
  for (Int_t im = 0; im < NModules; im++) {
    UInt_t tmask = (0xf << (im * NChPerModule)) & mTriggerMask;
    if (triggers_m & tmask) {
      for (UInt_t ic = 0; ic < NChPerModule; ic++) {
        mZDC.data[im][ic].f.Auto_m = 1;
      }
    }
    if (triggers_0 & tmask) {
      for (UInt_t ic = 0; ic < NChPerModule; ic++) {
        mZDC.data[im][ic].f.Auto_0 = 1;
      }
    }
    if (triggers_1 & tmask) {
      for (UInt_t ic = 0; ic < NChPerModule; ic++) {
        mZDC.data[im][ic].f.Auto_1 = 1;
      }
    }
    if (triggers_2 & tmask) {
      for (UInt_t ic = 0; ic < NChPerModule; ic++) {
        mZDC.data[im][ic].f.Auto_2 = 1;
      }
    }
    if (triggers_3 & tmask) {
      for (UInt_t ic = 0; ic < NChPerModule; ic++) {
        mZDC.data[im][ic].f.Auto_3 = 1;
      }
    }
  }
}

//______________________________________________________________________________
void Digits2Raw::insertLastBunch(int ibc, uint32_t orbit)
{

  // Orbit and bunch crossing identifiers
  UShort_t bc = 3563;

  // Reset scalers at orbit change
  if (orbit != mLastOrbit) {
    resetSums(orbit);
  }

  updatePedestalReference(bc);

  // Dummy bunch -> Do not increment scalers but reset output structure
  resetOutputStructure(bc, orbit, true);

  // Compute autotrigger bits and assign ALICE trigger bits
  assignTriggerBits(ibc, bc, orbit, true);

  // Insert payload for all channels
  for (Int_t im = 0; im < NModules; im++) {
    for (UInt_t ic = 0; ic < NChPerModule; ic++) {
      if (mModuleConfig->modules[im].readChannel[ic]) {
        auto id = mModuleConfig->modules[im].channelID[ic];
        auto base_m = mSimCondition->channels[id].pedestal;      // Average pedestal
        auto base_s = mSimCondition->channels[id].pedestalFluct; // Baseline oscillations
        auto base_n = mSimCondition->channels[id].pedestalNoise; // Electronic noise
        Double_t base = gRandom->Gaus(base_m, base_s);
        Int_t is = 0;
        Double_t val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s00 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        is++;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s01 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        is++;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s02 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        is++;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s03 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        is++;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s04 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        is++;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s05 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        is++;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s06 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        is++;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s07 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        is++;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s08 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        is++;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s09 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        is++;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s10 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
        is++;
        val = base + gRandom->Gaus(0, base_n);
        mZDC.data[im][ic].f.s11 = val < ADCMax ? (val > ADCMin ? val : ADCMin) : ADCMax;
      }
    }
  }
} // insertLastBunch

//______________________________________________________________________________
void Digits2Raw::convertDigits(int ibc)
{

  // Orbit and bunch crossing identifiers
  UShort_t bc = mBCD.ir.bc;
  UInt_t orbit = mBCD.ir.orbit;

  // Reset scalers at orbit change
  if (orbit != mLastOrbit) {
    resetSums(orbit);
  }

  updatePedestalReference(bc);

  // Not a dummy bunch -> Reset output structure and eventually flag hits and increment scalers
  resetOutputStructure(bc, orbit, false);

  // Compute autotrigger bits and assign ALICE trigger bits
  assignTriggerBits(ibc, bc, orbit, false);

  if (mVerbosity > 0) {
    mBCD.print();
    printf("Mask: %s\n", mPrintTriggerMask.data());
  }

  int chEnt = mBCD.ref.getFirstEntry();
  for (int ic = 0; ic < mBCD.ref.getEntries(); ic++) {
    const auto& chd = mzdcChData[chEnt++];
    if (mVerbosity > 0) {
      chd.print();
    }
    UShort_t bc = mBCD.ir.bc;
    UInt_t orbit = mBCD.ir.orbit;
    // Look for channel ID in digits and store channel (just one copy in output)
    // This is a limitation of software but we are not supposed to acquire the
    // same signal twice anyway
    for (Int_t im = 0; im < NModules; im++) {
      for (UInt_t ic = 0; ic < NChPerModule; ic++) {
        if (mModuleConfig->modules[im].channelID[ic] == chd.id &&
            mModuleConfig->modules[im].readChannel[ic]) {
          Int_t is = 0;
          mZDC.data[im][ic].f.s00 = chd.data[is];
          is++;
          mZDC.data[im][ic].f.s01 = chd.data[is];
          is++;
          mZDC.data[im][ic].f.s02 = chd.data[is];
          is++;
          mZDC.data[im][ic].f.s03 = chd.data[is];
          is++;
          mZDC.data[im][ic].f.s04 = chd.data[is];
          is++;
          mZDC.data[im][ic].f.s05 = chd.data[is];
          is++;
          mZDC.data[im][ic].f.s06 = chd.data[is];
          is++;
          mZDC.data[im][ic].f.s07 = chd.data[is];
          is++;
          mZDC.data[im][ic].f.s08 = chd.data[is];
          is++;
          mZDC.data[im][ic].f.s09 = chd.data[is];
          is++;
          mZDC.data[im][ic].f.s10 = chd.data[is];
          is++;
          mZDC.data[im][ic].f.s11 = chd.data[is];
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
  // Local interaction record (true and empty bunches)
  o2::InteractionRecord ir(mZDC.data[0][0].f.bc, mZDC.data[0][0].f.orbit);
  for (UInt_t im = 0; im < o2::zdc::NModules; im++) {
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
    if (tcond_triggered || (mIsContinuous && tcond_continuous) || (mZDC.data[im][0].f.bc == 3563)) {
      for (UInt_t ic = 0; ic < o2::zdc::NChPerModule; ic++) {
        if (mModuleConfig->modules[im].readChannel[ic]) {
          for (Int_t iw = 0; iw < o2::zdc::NWPerBc; iw++) {
            gsl::span<char> payload{reinterpret_cast<char*>(&mZDC.data[im][ic].w[iw][0]), data_size};
            mWriter.addData(mFeeID, mCruID, mLinkID, mEndPointID, ir, payload);
          }
        }
      }
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
        for (UInt_t ic = 0; ic < o2::zdc::NChPerModule; ic++) {
          if (mModuleConfig->modules[im].readChannel[ic]) {
            for (Int_t iw = 0; iw < o2::zdc::NWPerBc; iw++) {
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
void Digits2Raw::print_gbt_word(const UInt_t* word, const ModuleConfig* moduleConfig)
{
  if (word == nullptr) {
    printf("NULL\n");
    return;
  }
  unsigned __int128 val = word[2];
  val = val << 32;
  val = val | word[1];
  val = val << 32;
  val = val | word[0];
  static UInt_t last_orbit = 0, last_bc = 0;

  ULong64_t lsb = val;
  ULong64_t msb = val >> 64;
  UInt_t a = word[0];
  UInt_t b = word[1];
  UInt_t c = word[2];
  //UInt_t d=(msb>>32)&0xffffffff;
  //printf("\n%llx %llx ",lsb,msb);
  //printf("\n%8x %8x %8x %8x ",d,c,b,a);
  if ((a & 0x3) == 0) {
    UInt_t myorbit = (val >> 48) & 0xffffffff;
    UInt_t mybc = (val >> 36) & 0xfff;
    if (myorbit != last_orbit || mybc != last_bc) {
      printf("Orbit %9u bc %4u\n", myorbit, mybc);
      last_orbit = myorbit;
      last_bc = mybc;
    }
    printf("%04x %08x %08x ", c, b, a);
    UInt_t hits = (val >> 24) & 0xfff;
    Int_t offset = (lsb >> 8) & 0xffff - 32768;
    Float_t foffset = offset / 8.;
    UInt_t board = (lsb >> 2) & 0xf;
    UInt_t ch = (lsb >> 6) & 0x3;
    //printf("orbit %9u bc %4u hits %4u offset %+6i Board %2u Ch %1u", myorbit, mybc, hits, offset, board, ch);
    printf("orbit %9u bc %4u hits %4u offset %+8.3f Board %2u Ch %1u", myorbit, mybc, hits, foffset, board, ch);
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
    Short_t s[6];
    val = val >> 8;
    for (Int_t i = 0; i < 6; i++) {
      s[i] = val & 0xfff;
      if (s[i] > ADCMax) {
        s[i] = s[i] - ADCRange;
      }
      val = val >> 12;
    }
    printf(" %5d %5d %5d %5d %5d %5d", s[0], s[1], s[2], s[3], s[4], s[5]);
  } else if ((a & 0x3) == 2) {
    printf("%04x %08x %08x ", c, b, a);
    printf("%s %s %s %s %s %s ", a & 0x4 ? "H" : " ", a & 0x8 ? "TM" : "  ", a & 0x10 ? "T0" : "  ", a & 0x20 ? "T1" : "  ", a & 0x40 ? "T2" : "  ", a & 0x80 ? "T3" : "  ");
    printf("6-b ");
    Short_t s[6];
    val = val >> 8;
    for (Int_t i = 0; i < 6; i++) {
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
  const int LHCMaxBunches = o2::constants::lhc::LHCMaxBunches;
  mNEmpty = 0;
  for (Int_t ib = 0; ib < LHCMaxBunches; ib++) {
    Int_t mb = (ib + 31) % LHCMaxBunches; // beam gas from back of calorimeter
    Int_t m1 = (ib + 1) % LHCMaxBunches;  // previous bunch
    Int_t cb = ib;                        // current bunch crossing
    Int_t p1 = (ib - 1) % LHCMaxBunches;  // colliding + 1
    Int_t p2 = (ib + 1) % LHCMaxBunches;  // colliding + 2
    Int_t p3 = (ib + 1) % LHCMaxBunches;  // colliding + 3
    if (bunchPattern[mb] || bunchPattern[m1] || bunchPattern[cb] || bunchPattern[p1] || bunchPattern[p2] || bunchPattern[p3]) {
      mEmpty[ib] = mNEmpty;
    } else {
      mNEmpty++;
      mEmpty[ib] = mNEmpty;
    }
  }
  LOG(INFO) << "There are " << mNEmpty << " clean empty bunches";
}
