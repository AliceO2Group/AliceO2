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
// Trigger detector config needed here.
std::vector<CTPDigit> Digitizer::process(const gsl::span<o2::ctp::CTPInputDigit> detinputs)
{
  std::map<o2::detectors::DetID::ID, std::vector<CTPInput>> det2ctpinp = mCTPConfiguration->getDet2InputMap();
  std::map<std::string, uint64_t> detInputName2Mask = {{"V0A", 1}, {"V0B", 2}, {"T0A", 1}, {"T0A", 2}}; // To be taken from det database
  std::map<o2::InteractionRecord, std::vector<const CTPInputDigit*>> predigits;
  for (auto const& inp : detinputs) {
    predigits[inp.intRecord].push_back(&inp);
  }
  std::vector<CTPDigit> digits;
  for (auto const& hits : predigits) {
    CTPDigit data;
    data.intRecord = hits.first;
    std::bitset<CTP_NINPUTS> inpmaskcoll = 0;
    for (auto const inp : hits.second) {
      switch (inp->detector) {
        case o2::detectors::DetID::FT0: {
          // see dummy database above
          for (auto const& ctpinp : det2ctpinp[o2::detectors::DetID::FT0]) {
            uint64_t mask = (inp->inputsMask).to_ullong() & detInputName2Mask[ctpinp.name];
            inpmaskcoll |= std::bitset<CTP_NINPUTS>(mask);
          }
          break;
        }
        case o2::detectors::DetID::FV0: {
          for (auto const& ctpinp : det2ctpinp[o2::detectors::DetID::FV0]) {
            uint64_t mask = (inp->inputsMask).to_ullong() & detInputName2Mask[ctpinp.name];
            inpmaskcoll |= std::bitset<CTP_NINPUTS>(mask);
          }
          break;
        }
        default:
          // Error
          LOG(FATAL) << "CTP Digitizer: unknown detector:" << inp->detector;
          break;
      }
    }
    data.CTPInputMask = inpmaskcoll;
    calculateClassMask(inpmaskcoll, data.CTPClassMask);
    digits.emplace_back(data);
  }
  return std::move(digits);
}
void Digitizer::calculateClassMask(const std::bitset<CTP_NINPUTS> ctpinpmask, std::bitset<CTP_NCLASSES>& classmask)
{
  classmask = 0;
  for (auto const& tcl : mCTPConfiguration->getCTPClasses()) {
    if (tcl.descriptor->getInputsMask() & ctpinpmask.to_ullong()) {
      classmask |= (1 << tcl.classMask);
    }
  }
}
void Digitizer::init()
{
  // CTP Configuration
  if (mCCDBServer.empty()) {
    LOG(FATAL) << "CTP digitizer: CCDB server is not set";
  } else {
    LOG(INFO) << "CTP digitizer:: CCDB server:" << mCCDBServer;
  }
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(mCCDBServer);
  mCTPConfiguration = mgr.get<CTPConfiguration>(o2::ctp::CCDBPathCTPConfig);
  LOG(INFO) << " @@@ CTP Digitizer:: CCDB connected " << std::endl;
}
