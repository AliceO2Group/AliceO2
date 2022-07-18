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
#include "ZDCReconstruction/RecoConfigZDC.h"

using namespace o2::zdc;

void RecoConfigZDC::setSearch(uint32_t ich, int val)
{
  if (ich >= 0 && ich < NTDCChannels) {
    tdc_search[ich] = val;
  } else {
    LOG(fatal) << __func__ << " channel " << ich << " not in allowed range";
  }
}

int RecoConfigZDC::getSearch(uint32_t ich) const
{
  if (ich >= 0 && ich < NTDCChannels) {
    return tdc_search[ich];
  } else {
    LOG(fatal) << __func__ << " channel " << ich << " not in allowed range";
    return -1;
  }
}

void RecoConfigZDC::setIntegration(uint32_t ich, int beg, int end, int beg_ped, int end_ped)
{
  int sig_l = 0;
  int sig_h = NTimeBinsPerBC - 1;
  int ped_l = -NTimeBinsPerBC;
  int ped_h = NTimeBinsPerBC - 1;

  if (ich >= 0 && ich < NChannels) {
    if (beg < sig_l || beg > sig_h) {
      LOG(fatal) << "Integration start = " << beg << " for signal " << ich << " (" << ChannelNames[ich] << ") not in allowed range [" << sig_l << "-" << sig_h << "]";
      return;
    }
    if (end < sig_l || end > sig_h) {
      LOG(fatal) << "Integration end = " << beg << " for signal " << ich << " (" << ChannelNames[ich] << ") not in allowed range [" << sig_l << "-" << sig_h << "]";
      return;
    }
    if (end < beg) {
      LOG(fatal) << "Inconsistent integration range for signal " << ich << " (" << ChannelNames[ich] << "): [" << beg << "-" << end << "]";
      return;
    }
    if (beg_ped < ped_l || beg_ped > ped_h) {
      LOG(fatal) << "Pedestal integration start = " << beg_ped << " for signal " << ich << " (" << ChannelNames[ich] << ") not in allowed range [" << ped_l << "-" << ped_h << "]";
      return;
    }
    if (end_ped < ped_l || end_ped > ped_h) {
      LOG(fatal) << "Pedestal integration end = " << end_ped << " for signal " << ich << " (" << ChannelNames[ich] << ") not in allowed range [" << ped_l << "-" << ped_h << "]";
      return;
    }
    if (end_ped < beg_ped) {
      LOG(fatal) << "Inconsistent integration range for pedestal " << ich << " (" << ChannelNames[ich] << "): [" << beg_ped << "-" << end_ped << "]";
      return;
    }
    beg_int[ich] = beg;
    end_int[ich] = end;
    beg_ped_int[ich] = beg_ped;
    end_ped_int[ich] = end_ped;
  } else {
    LOG(fatal) << __func__ << " channel " << ich << " not in allowed range";
  }
}

void RecoConfigZDC::setPedThreshold(int32_t ich, float high, float low)
{
  if (ich >= 0 && ich < NChannels) {
    ped_thr_hi[ich] = high;
    ped_thr_lo[ich] = low;
  } else {
    LOG(fatal) << __func__ << " channel " << ich << " not in allowed range";
  }
}

void RecoConfigZDC::print() const
{
  LOGF(info, "RecoConfigZDC:%s%s%s%s",
       (low_pass_filter ? " LowPassFilter" : ""),
       (full_interpolation ? " FullInterpolation" : ""),
       (corr_signal ? " CorrSignal" : ""),
       (corr_background ? " CorrBackground" : ""));
  for (int itdc = 0; itdc < NTDCChannels; itdc++) {
    LOG(info) << itdc << " " << ChannelNames[TDCSignal[itdc]] << " search= " << tdc_search[itdc] << " = " << tdc_search[itdc] * FTDCVal << " ns";
  }
  for (Int_t ich = 0; ich < NChannels; ich++) {
    LOG(info) << ChannelNames[ich] << " integration: signal=[" << beg_int[ich] << ":" << end_int[ich] << "] pedestal=[" << beg_ped_int[ich] << ":" << end_ped_int[ich]
              << "] thresholds (" << ped_thr_hi[ich] << ", " << ped_thr_lo[ich] << ")";
  }
}
