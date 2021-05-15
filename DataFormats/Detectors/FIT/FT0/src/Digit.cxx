// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include <Framework/Logger.h>
#include <iostream>
#include <gsl/span>
#include <bitset>

using namespace o2::ft0;

void Triggers::printLog() const
{
  LOG(INFO) << "mTrigger: " << static_cast<uint16_t>(triggersignals);
  LOG(INFO) << "nChanA: " << static_cast<uint16_t>(nChanA) << " | nChanC: " << static_cast<uint16_t>(nChanC);
  LOG(INFO) << "amplA: " << amplA << " | amplC: " << amplC;
  LOG(INFO) << "timeA: " << timeA << " | timeC: " << timeC;
}

gsl::span<const ChannelData> Digit::getBunchChannelData(const gsl::span<const ChannelData> tfdata) const
{
  // extract the span of channel data for this bunch from the whole TF data
  return ref.getEntries() ? gsl::span<const ChannelData>(&tfdata[ref.getFirstEntry()], ref.getEntries()) : gsl::span<const ChannelData>();
}

void Digit::printStream(std::ostream& stream) const
{
  stream << "FT0 Digit:  BC " << mIntRecord.bc << " orbit " << mIntRecord.orbit << std::endl;
  stream << " A amp " << mTriggers.amplA << "  C amp " << mTriggers.amplC << " time A " << mTriggers.timeA << " time C " << mTriggers.timeC << " signals " << int(mTriggers.triggersignals) << std::endl;
}

std::ostream& operator<<(std::ostream& stream, const Digit& digi)
{
  digi.printStream(stream);
  return stream;
}
void Digit::printLog() const
{
  LOG(INFO) << "______________DIGIT DATA____________";
  LOG(INFO) << "BC: " << mIntRecord.bc << "| ORBIT: " << mIntRecord.orbit;
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