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

/// \file Digitizer.h
/// \author Roman Lietava

#ifndef ALICEO2_CTP_DIGITIZER_H
#define ALICEO2_CTP_DIGITIZER_H

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsCTP/Configuration.h"
#include "CCDB/BasicCCDBManager.h"
#include "CommonUtils/NameConf.h"
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
  void setCTPConfiguration(o2::ctp::CTPConfiguration* config);
  o2::ctp::CTPConfiguration* getDefaultCTPConfiguration();
  void init();

 private:
  // CTP configuration
  std::string mCCDBServer = o2::base::NameConf::getCCDBServer();
  CTPConfiguration* mCTPConfiguration = nullptr;
  ClassDefNV(Digitizer, 2);
};
} // namespace ctp
} // namespace o2
#endif // ALICEO2_CTP_DIGITIZER_H
