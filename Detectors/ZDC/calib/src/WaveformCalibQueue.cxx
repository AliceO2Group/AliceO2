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

#include "Framework/Logger.h"
#include "ZDCCalib/WaveformCalibQueue.h"

namespace o2
{
namespace zdc
{

uint32_t WaveformCalibQueue::append(RecEventFlat& ev)
{
  auto& toadd = ev.ir;
  // If queue is empty insert event
  if (mIR.size() == 0) {
    appendEv(ev);
    return 0;
  }
  // Check last element
  auto& last = mIR.back();
  // If BC are not consecutive, clear queue
  if (toadd.differenceInBC(last) > 1) {
    clear();
  }
  // If queue is not empty and is too long remove first element
  while (mIR.size() >= mN) {
    pop();
  }
  // If BC are consecutive or cleared queue append element
  appendEv(ev);
  if (mIR.size() == mN) {
    uint32_t mask = 0;
    for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
      // Check which channels satisfy the condition on TDC
      bool tdccond = true;
      for(int i = 0; i<mN; i++){
        int n = mNTDC[itdc].at(i);
        if(i==mPk){
          if(n!=1){
            tdccond = false;
            break;
          }
        }else{
          if(n!=0){
            tdccond = false;
            break;
          }
        }
      }
      if(tdccond){
        mask = mask | (0x1<<itdc);
      }
    }
    return mask;
  } else {
    return 0;
  }
}

void WaveformCalibQueue::appendEv(RecEventFlat& ev)
{
  LOG(info) << __func__ << " " << ev.ir.orbit << "." << ev.ir.bc;

  mIR.push_back(ev.ir);
  mEntry.push_back(ev.getNextEntry()-1);

  auto& curb = ev.getCurB();
  int firstw, nw;
  curb.getRefW(firstw, nw);
  mFirstW.push_back(firstw);
  mNW.push_back(nw);

  for (int ih = 0; ih < NH; ih++) {
    mHasInfos[ih].push_back(false);
  }
  if (ev.getNInfo() > 0) {
    // Need clean data (no messages)
    // We are sure there is no pile-up in any channel (too restrictive?)
    auto& decodedInfo = ev.getDecodedInfo();
    for (uint16_t info : decodedInfo) {
      uint8_t ch = (info >> 10) & 0x1f;
      uint16_t code = info & 0x03ff;
      auto& last = mHasInfos[SignalTDC[ch]].back();
      last = true;
    }
    // if (mVerbosity > DbgMinimal) {
    //   ev.print();
    // }
  }
  // NOTE: for the moment NH = NTDCChannels. If we remove this we will need to
  // have a mask of affected channels (towers)
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    int ich = o2::zdc::TDCSignal[itdc];
    int nhit = ev.NtdcV(itdc);
    if (ev.NtdcA(itdc) != nhit) {
      LOGF(error, "Mismatch in TDC %d data Val=%d Amp=%d\n", itdc, ev.NtdcV(itdc), ev.NtdcA(ich));
      mNTDC[itdc].push_back(0);
      mTDCA[itdc].push_back(0);
      mTDCP[itdc].push_back(0);
    } else if (nhit == 0) {
      mNTDC[itdc].push_back(0);
      mTDCA[itdc].push_back(0);
      mTDCP[itdc].push_back(0);
    } else {
      // Store single TDC entry
      mNTDC[itdc].push_back(nhit);
      mTDCA[itdc].push_back(ev.tdcA(itdc, 0));
      mTDCP[itdc].push_back(ev.tdcV(itdc, 0));
    }
  }
}

} // namespace zdc
} // namespace o2
