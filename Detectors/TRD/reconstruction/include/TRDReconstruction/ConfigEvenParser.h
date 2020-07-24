// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   EpnRawReaderTask.h
/// @author Sean Murray
/// @brief  TRD cru output data to tracklet task

#ifndef O2_TRD_EPNRAWREADER
#define O2_TRD_EPNRAWREADER

#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "TRDReconstruction/CruRawReader.h"
#include "TRDReconstruction/CompressedRawReader.h"
#include <fstream>

using namespace o2::framework;

namespace o2
{
namespace trd
{

class EpnRawReaderTask : public Task
{
 public:
  EpnRawReaderTask() = default;
  ~EpnRawReaderTask() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  CruRawReader mReader; // this will do the parsing, of raw data passed directly through the flp(no compression)
  //CompressedRawReader mCompressedReader;
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_CRU2TRACKLETTASK
