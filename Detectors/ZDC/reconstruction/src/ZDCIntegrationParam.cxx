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
#include "ZDCRaw/ZDCIntegrationParam.h"

using namespace o2::zdc;

void ZDCIntegrationParam::setIntegration(uint32_t ich, int beg, int end, int beg_ped, int end_ped)
{
  if (ich >= 0 && ich < NChannels) {
    if (beg < 0 || beg >= NTimeBinsPerBC) {
      LOG(FATAL) << "Integration start = " << beg << " for signal " << ich << " (" << ChannelNames[ich] << ") not in allowed range [0-" << NTimeBinsPerBC - 1;
      return;
    }
    if (end < 0 || end >= NTimeBinsPerBC) {
      LOG(FATAL) << "Integration end = " << end << " for signal " << ich << " (" << ChannelNames[ich] << ") not in allowed range [0-" << NTimeBinsPerBC - 1;
      return;
    }
    if (beg_ped < 0 || beg_ped >= NTimeBinsPerBC) {
      LOG(FATAL) << "Pedestal integration start = " << beg_ped << " for signal " << ich << " (" << ChannelNames[ich] << ") not in allowed range [0-" << NTimeBinsPerBC - 1;
      return;
    }
    if (beg < 0 || beg >= NTimeBinsPerBC) {
      LOG(FATAL) << "Pedestal integration end = " << end_ped << " for signal " << ich << " (" << ChannelNames[ich] << ") not in allowed range [0-" << NTimeBinsPerBC - 1;
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
    LOG(INFO) << ChannelNames[ich] << " integration: signal=[" << beg_int[ich] << "-" << end_int[ich] << "] pedestal=[" << beg_ped_int[ich] << "-" << end_ped_int[ich] <<"]";
  }
}
