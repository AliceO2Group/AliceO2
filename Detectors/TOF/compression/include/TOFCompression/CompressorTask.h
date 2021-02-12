// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompressorTask.h
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  TOF raw data compressor task

#ifndef O2_TOF_COMPRESSORTASK
#define O2_TOF_COMPRESSORTASK

#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "TOFCompression/Compressor.h"
#include <fstream>

using namespace o2::framework;

namespace o2
{
namespace tof
{

template <typename RDH, bool verbose>
class CompressorTask : public Task
{
 public:
  CompressorTask() = default;
  ~CompressorTask() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  Compressor<RDH, verbose> mCompressor;
  int mOutputBufferSize;
};

} // namespace tof
} // namespace o2

#endif /** O2_TOF_COMPRESSORTASK **/
