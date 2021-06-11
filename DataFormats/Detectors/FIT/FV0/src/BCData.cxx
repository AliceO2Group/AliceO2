// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFV0/BCData.h"
#include "DataFormatsFV0/ChannelData.h"
#include <bitset>

using namespace o2::fv0;

void Triggers::printLog() const
{
  LOG(INFO) << "mTrigger: " << static_cast<uint16_t>(triggerSignals);
  LOG(INFO) << "nChanA: " << static_cast<uint16_t>(nChanA) /* << " | nChanC: " << static_cast<uint16_t>(nChanC)*/;
  LOG(INFO) << "amplA: " << amplA /* << " | amplC: " << amplC*/;
  //  LOG(INFO) << "timeA: " << timeA << " | timeC: " << timeC;
}

void BCData::print() const
{
  ir.print();
  printf("\n");
}

gsl::span<const ChannelData> BCData::getBunchChannelData(const gsl::span<const ChannelData> tfdata) const
{
  // extract the span of channel data for this bunch from the whole TF data
  return ref.getEntries() ? gsl::span<const ChannelData>(&tfdata[ref.getFirstEntry()], ref.getEntries()) : gsl::span<const ChannelData>();
}
void BCData::printLog() const
{
  LOG(INFO) << "______________DIGIT DATA____________";
  LOG(INFO) << "BC: " << ir.bc << "| ORBIT: " << ir.orbit;
  LOG(INFO) << "Ref first: " << ref.getFirstEntry() << "| Ref entries: " << ref.getEntries();
  mTriggers.printLog();
}
void TriggersExt::printLog() const
{
  LOG(INFO) << "______________EXTENDED TRIGGERS____________";
  LOG(INFO) << "BC: " << mIntRecord.bc << "| ORBIT: " << mIntRecord.orbit;
  for (int i = 0; i < 20; i++) {
    LOG(INFO) << "N: " << i + 1 << " | TRG: " << mTriggerWords[i];
  }
}