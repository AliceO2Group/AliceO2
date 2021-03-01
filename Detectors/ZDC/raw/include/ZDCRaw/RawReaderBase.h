// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//file RawReaderBase.h base class for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
//Main purpuse is to decode ZDC data blocks and push them to DigitBlockFT0 for process

#ifndef ALICEO2_ZDC_RAWREADERBASE_H_
#define ALICEO2_ZDC_RAWREADERBASE_H_
#include <iostream>
#include <vector>
#include <map>

#include <Rtypes.h>
#include <CommonDataFormat/InteractionRecord.h>
#include "Headers/RAWDataHeader.h"
#include <Framework/Logger.h>
#include "DataFormatsZDC/RawEventData.h"
#include "ZDCSimulation/Digits2Raw.h"
#include "ZDCSimulation/SimCondition.h"

#include <gsl/span>
namespace o2
{
namespace zdc
{

class RawReaderBase
{
 public:
  RawReaderBase() = default;
  ~RawReaderBase() = default;

  std::map<InteractionRecord, EventData> mMapData; /// Raw data cache
  const ModuleConfig* mModuleConfig = nullptr;     /// Trigger/readout configuration object
  void setModuleConfig(const ModuleConfig* moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };

  //decoding binary data into data blocks
  EventChData mCh; // Channel data to be decoded
  int processWord(const uint32_t* word)
  {
    if (word == nullptr) {
      LOG(ERROR) << "NULL pointer";
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
        LOG(ERROR) << "Wrong word sequence";
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
        LOG(ERROR) << "Wrong word sequence";
      }
      mCh.f.fixed_0 = Id_wn;
      mCh.f.fixed_1 = Id_wn;
      mCh.f.fixed_2 = Id_wn;
    } else {
      // Word not present in payload
      LOG(FATAL) << "Event format error";
      return 1;
    }
    return 0;
  }

  void process(const EventChData& ch)
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

  //processing data blocks into digits
  void processBinaryData(gsl::span<const uint8_t> payload, int linkID)
  {
    size_t payloadSize = payload.size();
    for (int32_t ip = 0; ip < payloadSize; ip += 16) {
      //o2::zdc::Digits2Raw::print_gbt_word((const uint32_t*)&payload[ip]);
      processWord((const uint32_t*)&payload[ip]);
    }
  }
  /*
  void process(int linkID, gsl::span<const uint8_t> payload)
  {
    static_cast<RawReader*>(this)->processDigits(linkID,payload);
  }
  */
  //pop digits
  int getDigits(std::vector<BCData>& digitsBC, std::vector<ChannelData>& digitsCh, std::vector<PedestalData>& pedestalData)
  {
    if (mModuleConfig == nullptr) {
      LOG(FATAL) << "Missing ModuleConfig";
      return 0;
    }
    int bcCounter = mMapData.size();
    LOG(INFO) << "Processing #bc " << bcCounter;
    for (auto& [ir, ev] : mMapData) {
      // TODO: Error check
      // Pedestal data
      if (ir.bc == 3563) {
        auto& pdata = pedestalData.emplace_back();
        pdata.ir = ir;
        for (int32_t im = 0; im < NModules; im++) {
          for (int32_t ic = 0; ic < NChPerModule; ic++) {
            if (ev.data[im][ic].f.fixed_0 == Id_w0 && ev.data[im][ic].f.fixed_1 == Id_w1 && ev.data[im][ic].f.fixed_2 == Id_w2) {
              // Identify connected channel
              auto id = mModuleConfig->modules[im].channelID[ic];
              int offset = ev.data[im][ic].f.offset - 32768;
              pdata.data[id] = offset;
            } else if (ev.data[im][ic].f.fixed_0 == 0 && ev.data[im][ic].f.fixed_1 == 0 && ev.data[im][ic].f.fixed_2 == 0) {
              // Empty channel
            } else {
              LOG(ERROR) << "Data format error";
            }
          }
        }
      }
      // BC data
      auto& bcdata = digitsBC.emplace_back();
      bcdata.ir = ir;
      // Channel data
      bool inconsistent_event = false;
      bool filled_event = false;
      for (int32_t im = 0; im < NModules; im++) {
        ModuleTriggerMapData mt;
        mt.w = 0;
        bool filled_module = false;
        bool inconsistent_module = false;
        for (int32_t ic = 0; ic < NChPerModule; ic++) {
          if (ev.data[im][ic].f.fixed_0 == Id_w0 && ev.data[im][ic].f.fixed_1 == Id_w1 && ev.data[im][ic].f.fixed_2 == Id_w2) {
            auto& ch = ev.data[im][ic];
            UShort_t us[12];
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
              bcdata.triggers |= (0x1 << ((im - 1) * NChPerModule + ic));
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
          } else if (ev.data[im][ic].f.fixed_0 == 0 && ev.data[im][ic].f.fixed_1 == 0 && ev.data[im][ic].f.fixed_2 == 0) {
            // Empty channel
          } else {
            LOG(ERROR) << "Data format error";
          }
        }
        bcdata.moduleTriggers[im] = mt.w;
        if (inconsistent_module == true) {
          inconsistent_event = true;
        }
      }
      if (inconsistent_event) {
        LOG(ERROR) << "Inconsistent event";
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
};

} // namespace zdc
} // namespace o2

#endif
