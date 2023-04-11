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

#ifndef O2_TF_READER_SPEC_H
#define O2_TF_READER_SPEC_H

/// @file   TFReaderWorkflow.h

#include "Framework/WorkflowSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Headers/DataHeader.h"

namespace o2
{
namespace rawdd
{
struct TFReaderInp {
  std::string inpdata{};
  std::string detList{};
  std::string detListRawOnly{};
  std::string detListNonRawOnly{};
  std::string rawChannelConfig{};
  std::string copyCmd{};
  std::string tffileRegex{};
  std::string remoteRegex{};
  std::string metricChannel{};
  o2::detectors::DetID::mask_t detMask{};
  o2::detectors::DetID::mask_t detMaskRawOnly{};
  o2::detectors::DetID::mask_t detMaskNonRawOnly{};
  size_t minSHM = 0;
  int tfRateLimit = -999;
  int maxTFCache = 1;
  int maxFileCache = 1;
  int verbosity = 0;
  int64_t delay_us = 0;
  int maxLoops = 0;
  int maxTFs = -1;
  bool sendDummyForMissing = true;
  bool sup0xccdb = false;
  std::vector<o2::header::DataHeader> hdVec;
};

o2::framework::DataProcessorSpec getTFReaderSpec(o2::rawdd::TFReaderInp& rinp);

} // namespace rawdd
} // namespace o2
#endif
