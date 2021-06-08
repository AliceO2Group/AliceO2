// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FairLogger.h"
#include "ZDCReconstruction/ZDCIntegrationParam.h"

using namespace o2::zdc;

void ZDCIntegrationParam::setIntegration(uint32_t ich, int beg, int end, int beg_ped, int end_ped)
{
  int sig_l = 0;
  int sig_h = NTimeBinsPerBC - 1;
  int ped_l = -NTimeBinsPerBC;
  int ped_h = NTimeBinsPerBC - 1;

  if (ich >= 0 && ich < NChannels) {
    if (beg < sig_l || beg > sig_h) {
      LOG(FATAL) << "Integration start = " << beg << " for signal " << ich << " (" << ChannelNames[ich] << ") not in allowed range [" << sig_l << "-" << sig_h << "]";
      return;
    }
    if (end < sig_l || end > sig_h) {
      LOG(FATAL) << "Integration end = " << beg << " for signal " << ich << " (" << ChannelNames[ich] << ") not in allowed range [" << sig_l << "-" << sig_h << "]";
      return;
    }
    if (end < beg) {
      LOG(FATAL) << "Inconsistent integration range for signal " << ich << " (" << ChannelNames[ich] << "): [" << beg << "-" << end << "]";
      return;
    }
    if (beg_ped < ped_l || beg_ped > ped_h) {
      LOG(FATAL) << "Pedestal integration start = " << beg_ped << " for signal " << ich << " (" << ChannelNames[ich] << ") not in allowed range [" << ped_l << "-" << ped_h << "]";
      return;
    }
    if (end_ped < ped_l || end_ped > ped_h) {
      LOG(FATAL) << "Pedestal integration end = " << end_ped << " for signal " << ich << " (" << ChannelNames[ich] << ") not in allowed range [" << ped_l << "-" << ped_h << "]";
      return;
    }
    if (end_ped < beg_ped) {
      LOG(FATAL) << "Inconsistent integration range for pedestal " << ich << " (" << ChannelNames[ich] << "): [" << beg_ped << "-" << end_ped << "]";
      return;
    }
    beg_int[ich] = beg;
    end_int[ich] = end;
    beg_ped_int[ich] = beg_ped;
    end_ped_int[ich] = end_ped;
  } else {
    LOG(FATAL) << __func__ << " channel " << ich << " not in allowed range";
  }
}

void ZDCIntegrationParam::print()
{
  for (Int_t ich = 0; ich < NChannels; ich++) {
    LOG(INFO) << ChannelNames[ich] << " integration: signal=[" << beg_int[ich] << ":" << end_int[ich] << "] pedestal=[" << beg_ped_int[ich] << ":" << end_ped_int[ich] << "]";
  }
}
