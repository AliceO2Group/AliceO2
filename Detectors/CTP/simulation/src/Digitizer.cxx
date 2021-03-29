// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CTPSimulation/Digitizer.h"
#include "TRandom.h"
#include <cassert>
#include "FairLogger.h"
#include <bitset>

using namespace o2::ctp;

ClassImp(Digitizer);

void Digitizer::process(gsl::span<o2::ctp::CTPInputDigit> digits, gsl::span<o2::ctp::CTPRawData> rawdata)
{
  std::map<o2::InteractionRecord, std::vector<const CTPInputDigit*>> prerawdata;
  for (auto const& inp : digits) {
    if (prerawdata.count(inp.mIntRecord) == 0) {
      std::vector<const CTPInputDigit*> inputs;
      inputs.push_back(&inp);
      prerawdata[inp.mIntRecord] = inputs;
    } else
      prerawdata[inp.mIntRecord].push_back(&inp);
  }
  std::vector<CTPRawData> vrawdata;
  for (auto const& coll : prerawdata) {
    CTPRawData data;
    data.mIntRecord = coll.first;
    std::bitset<CTP_NINPUTS> inpmaskcoll = 0;
    for (auto const inp : coll.second) {
      switch (inp->mDetector) {
        case o2::detectors::DetID::FT0: {
          std::bitset<CTP_NINPUTS> inpmask = std::bitset<CTP_NINPUTS>(
            (inp->mInputsMask & CTP_INPUTMASK_FT0.second).to_ullong());
          inpmaskcoll |= inpmask << CTP_INPUTMASK_FT0.first;
          break;
        }
        case o2::detectors::DetID::FV0: {
          std::bitset<CTP_NINPUTS> inpmask = std::bitset<CTP_NINPUTS>(
            (inp->mInputsMask & CTP_INPUTMASK_FT0.second).to_ullong());
          inpmaskcoll |= inpmask << CTP_INPUTMASK_FT0.first;
          break;
        }
        default:
          // Error
          LOG(ERROR) << "CTP Digitizer: unknown detector:" << inp->mDetector;
          break;
      }
    }
    data.mCTPInputMask = inpmaskcoll;
    calculateClassMask(coll.second, data.mCTPClassMask);
    vrawdata.emplace_back(data);
  }
  rawdata = gsl::span<CTPRawData>(vrawdata);
}
void Digitizer::calculateClassMask(std::vector<const CTPInputDigit*> inputs, std::bitset<CTP_NCLASSES>& classmask)
{
  // Example of Min Bias as V0 or T0
  classmask = 0;
  if (inputs.size() > 1) {
    classmask = 1;
  }
}
void Digitizer::init()
{
  LOG(INFO) << " @@@ CTP Digitizer::init. Nothing done. " << std::endl;
}
