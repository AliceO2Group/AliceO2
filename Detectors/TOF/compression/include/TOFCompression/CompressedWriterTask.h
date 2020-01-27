// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompressedWriterTask.h
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  TOF compressed data writer task

#ifndef O2_TOF_COMPRESSEDWRITERTASK
#define O2_TOF_COMPRESSEDWRITERTASK

#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include <fstream>

using namespace o2::framework;

namespace o2
{
namespace tof
{

class CompressedWriterTask : public Task
{
 public:
  CompressedWriterTask() = default;
  ~CompressedWriterTask() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

  static DataProcessorSpec getSpec();

 private:
  bool mStatus = false;
  std::ofstream mFile;
};

} // namespace tof
} // namespace o2

#endif /* O2_TOF_COMPRESSEDWRITERTASK */
