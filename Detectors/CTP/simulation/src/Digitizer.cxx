// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digitizer.cxx
/// \author Roman Lietava

#include "CTPSimulation/Digitizer.h"
#include "TRandom.h"
#include <cassert>
#include "FairLogger.h"
#include <bitset>

using namespace o2::ctp;

ClassImp(Digitizer);

std::vector<CTPDigit> Digitizer::process(const gsl::span<o2::ctp::CTPInputDigit> digits)
{
  std::map<o2::InteractionRecord, std::vector<const CTPInputDigit*>> prerawdata;
  for (auto const& inp : digits) {
    prerawdata[inp.intRecord].push_back(&inp);
  }
  std::vector<CTPDigit> vrawdata;
  for (auto const& coll : prerawdata) {
    CTPDigit data;
    data.intRecord = coll.first;
    std::bitset<CTP_NINPUTS> inpmaskcoll = 0;
    for (auto const inp : coll.second) {
      switch (inp->detector) {
        case o2::detectors::DetID::FT0: {
          std::bitset<CTP_NINPUTS> inpmask = std::bitset<CTP_NINPUTS>(
            (inp->inputsMask & CTP_INPUTMASK_FT0.second).to_ullong());
          inpmaskcoll |= inpmask << CTP_INPUTMASK_FT0.first;
          break;
        }
        case o2::detectors::DetID::FV0: {
          std::bitset<CTP_NINPUTS> inpmask = std::bitset<CTP_NINPUTS>(
            (inp->inputsMask & CTP_INPUTMASK_FV0.second).to_ullong());
          inpmaskcoll |= inpmask << CTP_INPUTMASK_FV0.first;
          break;
        }
        default:
          // Error
          LOG(ERROR) << "CTP Digitizer: unknown detector:" << inp->detector;
          break;
      }
    }
    data.CTPInputMask = inpmaskcoll;
    calculateClassMask(coll.second, data.CTPClassMask);
    vrawdata.emplace_back(data);
  }
  return std::move(vrawdata);
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
