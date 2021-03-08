// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_READRAWFILESPEC_H_
#define DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_READRAWFILESPEC_H_

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "HMPIDBase/Common.h"
#include "HMPIDBase/Digit.h"

namespace o2
{
namespace hmpid
{

class RawFileReaderTask : public framework::Task
{
 public:
  RawFileReaderTask() = default;
  ~RawFileReaderTask() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;

 private:
  std::ifstream mInputFile{}; ///< input file
  bool mPrint = false;        ///< print debug messages

  ExecutionTimer mExTimer;
};

o2::framework::DataProcessorSpec getReadRawFileSpec(std::string inputSpec = "HMP/RAWDATA");

} // end namespace hmpid
} // end namespace o2

#endif /* DETECTORS_HMPID_WORKFLOW_INCLUDE_HMPIDWORKFLOW_READRAWFILESPEC_H_ */
