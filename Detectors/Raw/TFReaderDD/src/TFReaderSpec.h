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

namespace o2
{
namespace rawdd
{
struct TFReaderInp {
  std::string inpdata{};
  std::string detList{};
  std::string rawChannelConfig{};
  std::string copyCmd{};
  std::string tffileRegex{};
  std::string remoteRegex{};
  int maxTFCache = 1;
  int maxFileCache = 1;
  int verbosity = 0;
  int64_t delay_us = 0;
  int maxLoops = 0;
  int maxTFs = -1;
};

o2::framework::DataProcessorSpec getTFReaderSpec(o2::rawdd::TFReaderInp& rinp);

} // namespace rawdd
} // namespace o2
#endif
