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

/// @file   TOFRawWriterSpec.h

#ifndef O2_TOF_RAWWRITER_H
#define O2_TOF_RAWWRITER_H

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <string>
#include "TOFReconstruction/Encoder.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

class RawWriter : public Task
{
 public:
  RawWriter() = default;
  ~RawWriter() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  std::string mOutFileName; // read from workflow
  std::string mOutDirName;  // read from workflow
  std::string mFileFor;     // output granularity
  bool mOldFormat = false;  // encode with old format (zeros in word 2 and 3)
};

/// create a processor spec
/// write TOF raw file
o2::framework::DataProcessorSpec getTOFRawWriterSpec();

} // namespace tof
} // namespace o2

#endif /* O2_TOF_RAWWRITER_H */
