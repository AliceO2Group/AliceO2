// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   EpnDataReaderTask.h
/// @author Sean Murray
/// @brief  TRD epn task to read incoming data

#ifndef O2_TRD_EPNDATAREADERTASK
#define O2_TRD_EPNDATAREADERTASK

#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "TRDRaw/CruRawReader.h"
#include "TRDRaw/CompressedRawReader.h"
#include <fstream>

using namespace o2::framework;

namespace o2
{
namespace trd
{

class EpnDataReaderTask : public Task
{
 public:
  EpnDataReaderTask() = default;
  ~EpnDataReaderTask() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  CruRawReader mReader;                  // this will do the parsing, of raw data passed directly through the flp(no compression)
  CompressedRawReader mCompressedReader; //this will handle the incoming compressed data from the flp
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_EPNRAWREADERTASK
