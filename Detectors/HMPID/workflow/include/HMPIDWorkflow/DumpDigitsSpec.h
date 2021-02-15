// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    DatDecoderSpec.h
/// \author  Andrea Ferrero
///
/// \brief Definition of a data processor to run the raw decoding
///

#ifndef DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_DUMPDIGITSPEC_H_
#define DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_DUMPDIGITSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "HMPIDBase/Common.h"
#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace hmpid
{

class DumpDigitsTask : public framework::Task
{
 public:
  DumpDigitsTask() = default;
  ~DumpDigitsTask() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) override;

 private:
  bool mPrintDigits = false;
  bool mIsOutputOnFile = false;
  std::string mOutputFileName = "";
  std::ofstream mOsFile;

  int mOrbit;
  int mBc;
  long mDigitsReceived;

  ExecutionTimer mExTimer;
};

o2::framework::DataProcessorSpec getDumpDigitsSpec(std::string inputSpec = "HMP/DIGITS");

} // end namespace hmpid
} // end namespace o2

#endif
