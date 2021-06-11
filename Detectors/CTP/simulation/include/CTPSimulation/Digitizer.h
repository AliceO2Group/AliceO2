// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digitizer.h
/// \author Roman Lietava

#ifndef ALICEO2_CTP_DIGITIZER_H
#define ALICEO2_CTP_DIGITIZER_H

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsCTP/Configuration.h"
#include "CCDB/BasicCCDBManager.h"
#include <gsl/span>

namespace o2
{
namespace ctp
{
class Digitizer
{
 public:
  Digitizer() = default;
  ~Digitizer() = default;
  void setCCDBServer(const std::string& server) { mCCDBServer = server; }
  std::vector<CTPDigit> process(const gsl::span<o2::ctp::CTPInputDigit> detinputs);
  void calculateClassMask(const std::bitset<CTP_NINPUTS> ctpinpmask, std::bitset<CTP_NCLASSES>& classmask);
  void init();
 private:
  // CTP configuration
  std::string mCCDBServer = "http://ccdb-test.cern.ch:8080";
  CTPConfiguration* mCTPConfiguration = nullptr;
  ClassDefNV(Digitizer, 2);
};
} // namespace ctp
} // namespace o2
#endif //ALICEO2_CTP_DIGITIZER_H
