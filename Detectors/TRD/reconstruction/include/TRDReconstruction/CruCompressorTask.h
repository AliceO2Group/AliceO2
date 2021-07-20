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

/// @file   CruCompressorTask.h
/// @author Sean Murray
/// @brief  TRD cru output data to tracklet task

#ifndef O2_TRD_CRU2TRACKLETTASK
#define O2_TRD_CRU2TRACKLETTASK

#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "TRDReconstruction/CruRawReader.h"
#include <fstream>

using namespace o2::framework;

namespace o2
{
namespace trd
{

class CruCompressorTask : public Task
{
 public:
  CruCompressorTask() = default;
  ~CruCompressorTask() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  uint64_t buildEventOutput();

 private:
  CruRawReader mReader; // this will do the parsing.
  std::array<uint64_t, o2::trd::constants::HBFBUFFERMAX> mOutBuffer;
  bool mVerbose{false};
  bool mDataVerbose{false};
  bool mHeaderVerbose{false};
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_CRU2TRACKLETTASK
