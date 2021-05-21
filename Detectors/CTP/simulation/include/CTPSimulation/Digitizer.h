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
  std::vector<CTPDigit> process(const gsl::span<o2::ctp::CTPInputDigit> digits);
  void calculateClassMask(std::vector<const CTPInputDigit*> inputs, std::bitset<CTP_NCLASSES>& classmask);
  void init();

 private:
  ClassDefNV(Digitizer, 1);
};
} // namespace ctp
} // namespace o2
#endif //ALICEO2_CTP_DIGITIZER_H
