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

/// \file Digitizer.cxx
/// \author Roman Lietava

#include "CTPSimulation/Digitizer.h"
#include "TRandom.h"
#include <cassert>
#include <fairlogger/Logger.h>
#include <bitset>

using namespace o2::ctp;

ClassImp(Digitizer);
// Trigger detector config needed here.
std::vector<CTPDigit> Digitizer::process(const gsl::span<o2::ctp::CTPInputDigit> detinputs)
{
  std::map<o2::detectors::DetID::ID, std::vector<CTPInput>> det2ctpinp = mCTPConfiguration->getDet2InputMap();
  // To be taken from config database ?
  std::map<std::string, uint64_t> detInputName2Mask =
    {{"MVBA", 1}, {"MVOR", 2}, {"MVNC", 4}, {"MVCH", 8}, {"MVIR", 0x10}, {"MT0A", 1}, {"MT0C", 2}, {"MTSC", 4}, {"MTCE", 8}, {"MTVX", 0x10}, {"0U0A", 1}, {"0U0C", 2}, {"0USC", 4}, {"0UCE", 8}, {"0UVX", 0x10}, {"MBA", 0x1}, {"0EMC", 0x2}, {"0DMC", 0x4}};

  // pre-sorting detector inputs per interaction record
  std::map<o2::InteractionRecord, std::vector<const CTPInputDigit*>> predigits;
  for (auto const& inp : detinputs) {
    predigits[inp.intRecord].push_back(&inp);
  }

  std::vector<CTPDigit> digits;
  for (auto const& hits : predigits) {
    std::bitset<CTP_NINPUTS> inpmaskcoll = 0;
    for (auto const inp : hits.second) {
      switch (inp->detector) {
        case o2::detectors::DetID::FT0: {
          // see dummy database above
          for (auto const& ctpinp : det2ctpinp[o2::detectors::DetID::FT0]) {
            uint64_t mask = (inp->inputsMask).to_ullong() & detInputName2Mask[ctpinp.name];
            if (mask) {
              inpmaskcoll |= std::bitset<CTP_NINPUTS>(ctpinp.inputMask);
            }
          }
          break;
        }
        case o2::detectors::DetID::FV0: {
          for (auto const& ctpinp : det2ctpinp[o2::detectors::DetID::FV0]) {
            uint64_t mask = (inp->inputsMask).to_ullong() & detInputName2Mask[ctpinp.name];
            if (mask) {
              inpmaskcoll |= std::bitset<CTP_NINPUTS>(ctpinp.inputMask);
            }
          }
          break;
        }
        case o2::detectors::DetID::FDD: {
          for (auto const& ctpinp : det2ctpinp[o2::detectors::DetID::FDD]) {
            uint64_t mask = (inp->inputsMask).to_ullong() & detInputName2Mask[ctpinp.name];
            if (mask) {
              inpmaskcoll |= std::bitset<CTP_NINPUTS>(ctpinp.inputMask);
            }
          }
          break;
        }
        case o2::detectors::DetID::EMC: {
          uint64_t inpmaske = inp->inputsMask.to_ullong();
          if (inpmaske & detInputName2Mask["MBA"]) {
            inpmaskcoll |= std::bitset<CTP_NINPUTS>(CTP_NINPUTS - 1);
            inpmaske &= ~(1ull < detInputName2Mask["MBA"]);
          }
          for (auto const& ctpinp : det2ctpinp[o2::detectors::DetID::EMC]) {
            uint64_t mask = inpmaske & detInputName2Mask[ctpinp.name];
            if (mask) {
              inpmaskcoll |= std::bitset<CTP_NINPUTS>(ctpinp.inputMask);
            }
          }
          break;
        }
        case o2::detectors::DetID::PHS: {
          for (auto const& ctpinp : det2ctpinp[o2::detectors::DetID::PHS]) {
            uint64_t mask = (inp->inputsMask).to_ullong() & detInputName2Mask[ctpinp.name];
            if (mask) {
              inpmaskcoll |= std::bitset<CTP_NINPUTS>(ctpinp.inputMask);
            }
          }
          break;
        }
        case o2::detectors::DetID::ZDC: {
          for (auto const& ctpinp : det2ctpinp[o2::detectors::DetID::ZDC]) {
            uint64_t mask = (inp->inputsMask).to_ullong() & detInputName2Mask[ctpinp.name];
            if (mask) {
              inpmaskcoll |= std::bitset<CTP_NINPUTS>(ctpinp.inputMask);
            }
          }
          break;
        }
        default:
          // Error
          LOG(error) << "CTP Digitizer: unknown detector:" << inp->detector;
          break;
      }
    } // end loop over trigger input for this interaction
    if (inpmaskcoll.to_ullong()) {
      // we put the trigger only when non-trivial
      CTPDigit data;
      data.intRecord = hits.first;
      data.CTPInputMask = inpmaskcoll;
      calculateClassMask(inpmaskcoll, data.CTPClassMask);
      digits.emplace_back(data);
      LOG(info) << "Trigger-Event " << data.intRecord.bc << " " << data.intRecord.orbit << " Input mask:" << inpmaskcoll;
    }
  }
  return std::move(digits);
}
void Digitizer::calculateClassMask(const std::bitset<CTP_NINPUTS> ctpinpmask, std::bitset<CTP_NCLASSES>& classmask)
{
  classmask = 0;
  for (auto const& tcl : mCTPConfiguration->getCTPClasses()) {
    bool tvxMBemc = tcl.name.find("C0TVX-B-NOPF-EMC") != std::string::npos;
    tvxMBemc != tcl.name.find("C0TVX-A-NOPF-EMC") != std::string::npos;
    tvxMBemc != tcl.name.find("C0TVX-C-NOPF-EMC") != std::string::npos;
    tvxMBemc != tcl.name.find("C0TVX-E-NOPF-EMC") != std::string::npos;
    tvxMBemc != tcl.name.find("minbias_TVX_L0") != std::string::npos; // 2022
    if (tvxMBemc && ctpinpmask[CTP_NINPUTS - 1]) {
      classmask |= (1 << tcl.classMask);
      LOG(info) << "adding MBA:" << tcl.name;
    } else {
      if (tcl.descriptor->getInputsMask() & ctpinpmask.to_ullong()) {
        classmask |= (1 << tcl.classMask);
      }
    }
  }
  LOG(info) << "class mask:" << classmask;
}
void Digitizer::init()
{
  // CTP Configuration
  if (mCCDBServer.empty()) {
    LOG(fatal) << "CTP digitizer: CCDB server is not set";
  } else {
    LOG(info) << "CTP digitizer:: CCDB server:" << mCCDBServer;
  }
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(mCCDBServer);
  map<string, string> metadata = {};
  long timestamp = 1546300800000;
  mCTPConfiguration = mgr.getSpecific<CTPConfiguration>(o2::ctp::CCDBPathCTPConfig, timestamp, metadata);
  mCTPConfiguration->printStream(std::cout);
  LOG(info) << " @@@ CTP Digitizer:: CCDB connected " << std::endl;
}
