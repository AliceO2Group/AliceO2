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

#ifndef O2_MCHIO_DIGIT_WRITER_SPEC_H_
#define O2_MCHIO_DIGIT_WRITER_SPEC_H_

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace mch
{
o2::framework::DataProcessorSpec getMCHDigitWriterSpec(bool mctruth);

o2::framework::DataProcessorSpec getDigitWriterSpec(
  bool useMC,
  std::string_view specName = "mch-digit-writer",
  std::string_view outfile = "mchdigits.root",
  std::string_view inputDigitDataDescription = "DIGITS",
  std::string_view inputDigitRofDataDescription = "DIGITROFS");

} // end namespace mch
} // end namespace o2

#endif
