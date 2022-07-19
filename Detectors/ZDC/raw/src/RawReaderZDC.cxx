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

#include "ZDCRaw/RawReaderZDC.h"

namespace o2
{
namespace zdc
{

void RawReaderZDC::clear()
{
  mDigitsBC.clear();
  mDigitsCh.clear();
  mOrbitData.clear();
}

void RawReaderZDC::processBinaryData(gsl::span<const uint8_t> payload, int linkID)
{
  if (0 <= linkID && linkID < 16) {
    size_t payloadSize = payload.size();
    for (int32_t ip = 0; ip < payloadSize; ip += 16) {
      // o2::zdc::Digits2Raw::print_gbt_word((const uint32_t*)&payload[ip]);
      processWord((const uint32_t*)&payload[ip]);
    }
  } else {
    // put here code in case of bad rdh.linkID value
    LOG(info) << "WARNING! WRONG LINK ID! " << linkID;
    return;
  }
}

int RawReaderZDC::processWord(const uint32_t* word)
{
  if (word == nullptr) {
    LOG(error) << "NULL pointer";
    return 1;
  }
  if ((word[0] & 0x3) == Id_w0) {
    for (int32_t iw = 0; iw < NWPerGBTW; iw++) {
      mCh.w[0][iw] = word[iw];
    }
  } else if ((word[0] & 0x3) == Id_w1) {
    if (mCh.f.fixed_0 == Id_w0) {
      for (int32_t iw = 0; iw < NWPerGBTW; iw++) {
        mCh.w[1][iw] = word[iw];
      }
    } else {
      LOG(error) << "Wrong word sequence";
      mCh.f.fixed_0 = Id_wn;
      mCh.f.fixed_1 = Id_wn;
      mCh.f.fixed_2 = Id_wn;
    }
  } else if ((word[0] & 0x3) == Id_w2) {
    if (mCh.f.fixed_0 == Id_w0 && mCh.f.fixed_1 == Id_w1) {
      for (int32_t iw = 0; iw < NWPerGBTW; iw++) {
        mCh.w[2][iw] = word[iw];
      }
      process(mCh);
    } else {
      LOG(error) << "Wrong word sequence";
    }
    mCh.f.fixed_0 = Id_wn;
    mCh.f.fixed_1 = Id_wn;
    mCh.f.fixed_2 = Id_wn;
  } else {
    // Word not present in payload
    LOG(fatal) << "Event format error";
    return 1;
  }
  return 0;
}

void RawReaderZDC::process(const EventChData& ch)
{
  InteractionRecord ir(ch.f.bc, ch.f.orbit);
  auto& mydata = mMapData[ir];
  int32_t im = ch.f.board;
  int32_t ic = ch.f.ch;
  for (int32_t iwb = 0; iwb < NWPerBc; iwb++) {
    for (int32_t iwg = 0; iwg < NWPerGBTW; iwg++) {
      mydata.data[im][ic].w[iwb][iwg] = mCh.w[iwb][iwg];
    }
  }
}

// pop digits
int RawReaderZDC::getDigits(std::vector<BCData>& digitsBC, std::vector<ChannelData>& digitsCh, std::vector<OrbitData>& orbitData)
{
  if (mModuleConfig == nullptr) {
    LOG(fatal) << "Missing ModuleConfig";
    return 0;
  }
  union {
    uint16_t uns;
    int16_t sig;
  } word16;
  int bcCounter = mMapData.size();
  LOG(info) << "Processing #bc " << bcCounter;
  for (auto& [ir, ev] : mMapData) {
    // TODO: Error check
    // Pedestal data
    if (ir.bc == 3563) {
      auto& pdata = orbitData.emplace_back();
      pdata.ir = ir;
      for (int32_t im = 0; im < NModules; im++) {
        for (int32_t ic = 0; ic < NChPerModule; ic++) {
          if (ev.data[im][ic].f.fixed_0 == Id_w0 && ev.data[im][ic].f.fixed_1 == Id_w1 && ev.data[im][ic].f.fixed_2 == Id_w2) {
            // Protection for channels that are not supposed to readout but may be present in payload
            // just for scaler and pedestal readout at end of orbit.
            if (mModuleConfig->modules[im].readChannel[ic]) {
              // Identify connected channel
              auto id = mModuleConfig->modules[im].channelID[ic];
              word16.uns = ev.data[im][ic].f.offset;
              pdata.data[id] = word16.sig;
              if (ev.data[im][ic].f.dLoss) {
                // Produce a scaler overflow to signal a problem
                // Most significant bit indicates data loss
                // Default initializer 0x8fff will indicate that orbit data is lost
                pdata.scaler[id] = ev.data[im][ic].f.hits | 0x8000;
              } else {
                pdata.scaler[id] = ev.data[im][ic].f.hits;
              }
            }
          } else if (ev.data[im][ic].f.fixed_0 == 0 && ev.data[im][ic].f.fixed_1 == 0 && ev.data[im][ic].f.fixed_2 == 0) {
            // Empty channel
          } else {
            LOG(error) << "Data format error";
          }
        }
      }
    }
    // BC data
    auto& bcdata = digitsBC.emplace_back();
    bcdata.ir = ir;
    // An inconsistent event has as at least one inconsistent module
    bool inconsistent_event = false;
    bool filled_event = false;
    bcdata.ref.setFirstEntry(digitsCh.size());
    uint32_t ncd = 0;
    // Channel data
    for (int32_t im = 0; im < NModules; im++) {
      ModuleTriggerMapData mt;
      mt.w = 0;
      bool filled_module = false;
      bool inconsistent_module = false;
      for (int32_t ic = 0; ic < NChPerModule; ic++) {
        // Check if payload is present for channel
        if (ev.data[im][ic].f.fixed_0 == Id_w0 && ev.data[im][ic].f.fixed_1 == Id_w1 && ev.data[im][ic].f.fixed_2 == Id_w2) {
          bcdata.channels |= 0x1 << (NChPerModule * im + ic); // Flag channel as present
          auto& ch = ev.data[im][ic];
          uint16_t us[12];
          us[0] = ch.f.s00;
          us[1] = ch.f.s01;
          us[2] = ch.f.s02;
          us[3] = ch.f.s03;
          us[4] = ch.f.s04;
          us[5] = ch.f.s05;
          us[6] = ch.f.s06;
          us[7] = ch.f.s07;
          us[8] = ch.f.s08;
          us[9] = ch.f.s09;
          us[10] = ch.f.s10;
          us[11] = ch.f.s11;
          // Identify connected channel
          auto& chd = digitsCh.emplace_back();
          auto id = mModuleConfig->modules[im].channelID[ic];
          chd.id = id;
          for (int32_t is = 0; is < NTimeBinsPerBC; is++) {
            if (us[is] > ADCMax) {
              chd.data[is] = us[is] - ADCRange;
            } else {
              chd.data[is] = us[is];
            }
          }
          // Trigger bits
          if (ch.f.Hit) {
            bcdata.triggers |= (0x1 << (im * NChPerModule + ic));
          }
          // TODO: Alice trigger bits
          // TODO: consistency checks
          if (filled_event == false) {
            mt.f.Alice_0 = ch.f.Alice_0;
            mt.f.Alice_1 = ch.f.Alice_1;
            mt.f.Alice_2 = ch.f.Alice_2;
            mt.f.Alice_3 = ch.f.Alice_3;
            filled_event = true;
          } else if (mt.f.Alice_0 != ch.f.Alice_0 || mt.f.Alice_1 != ch.f.Alice_1 || mt.f.Alice_2 != ch.f.Alice_2 || mt.f.Alice_3 != ch.f.Alice_3) {
            inconsistent_event = true;
          }
          if (filled_module == false) {
            mt.f.Auto_m = ch.f.Auto_m;
            mt.f.Auto_0 = ch.f.Auto_0;
            mt.f.Auto_1 = ch.f.Auto_1;
            mt.f.Auto_2 = ch.f.Auto_2;
            mt.f.Auto_3 = ch.f.Auto_3;
            filled_module = true;
          } else if (mt.f.Auto_m != ch.f.Auto_m || mt.f.Auto_0 != ch.f.Auto_0 || mt.f.Auto_1 != ch.f.Auto_1 || mt.f.Auto_2 != ch.f.Auto_2 || mt.f.Auto_3 != ch.f.Auto_3) {
            inconsistent_module = true;
          }
          // Verify trigger condition (if requested)
          if (mVerifyTrigger) {
            if ((mt.f.Alice_0 || mt.f.Alice_1) || (mt.f.Alice_2 && (mt.f.Auto_0 || mt.f.Auto_m)) || (mt.f.Alice_3 && mt.f.Auto_0) || (mt.f.Auto_0 || mt.f.Auto_1)) {
              ncd++;
            } else {
              digitsCh.pop_back();
            }
          } else {
            ncd++;
          }
        } else if (ev.data[im][ic].f.fixed_0 == 0 && ev.data[im][ic].f.fixed_1 == 0 && ev.data[im][ic].f.fixed_2 == 0) {
          // Empty channel
        } else {
          LOG(error) << "Data format error";
        }
      }
      bcdata.moduleTriggers[im] = mt.w;
      if (inconsistent_module == true) {
        inconsistent_event = true;
      }
    }
    if (ncd == 0) {
      // Remove empty orbits (keep pedestal information)
      digitsBC.pop_back();
    } else {
      bcdata.ref.setEntries(ncd);
      if (mDumpData) {
        bcdata.print(mTriggerMask);
        auto first_entry = bcdata.ref.getFirstEntry();
        for (Int_t icd = 0; icd < ncd; icd++) {
          digitsCh[icd + first_entry].print();
        }
      }
    }
    if (inconsistent_event) {
      LOG(error) << "Inconsistent event";
      for (int32_t im = 0; im < NModules; im++) {
        for (int32_t ic = 0; ic < NChPerModule; ic++) {
          if (ev.data[im][ic].f.fixed_0 == Id_w0 && ev.data[im][ic].f.fixed_1 == Id_w1 && ev.data[im][ic].f.fixed_2 == Id_w2) {
            for (int32_t iw = 0; iw < NWPerBc; iw++) {
              o2::zdc::Digits2Raw::print_gbt_word((const uint32_t*)&ev.data[im][ic].w[iw][0]);
            }
          }
        }
      }
    }
  }

  mMapData.clear();
  return bcCounter;
}

//______________________________________________________________________________
void RawReaderZDC::setTriggerMask()
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
#ifdef O2_ZDC_DEBUG
    uint32_t mytmask = mTriggerMask >> (im * NChPerModule);
    LOGF(info, "Trigger mask for module %d 0123 %c%c%c%c", im, mytmask & 0x1 ? 'T' : 'N', mytmask & 0x2 ? 'T' : 'N', mytmask & 0x4 ? 'T' : 'N', mytmask & 0x8 ? 'T' : 'N');
#endif
  }
  LOGF(info, "trigger_mask=0x%08x %s", mTriggerMask, printTriggerMask.c_str());
}
} // namespace zdc
} // namespace o2
