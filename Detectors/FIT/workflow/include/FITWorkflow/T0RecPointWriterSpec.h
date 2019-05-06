// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   T0RecPointWriterSpec.h

#ifndef O2_FIT_T0RECPOINTWRITER_H
#define O2_FIT_T0RECPOINTWRITER_H

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

using namespace o2::framework;

namespace o2
{
namespace t0
{

class T0RecPointWriter : public Task
{
 public:
  T0RecPointWriter(bool useMC = true) : mUseMC(useMC) {}
  ~T0RecPointWriter() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  bool mFinished = false;
  bool mUseMC = true;

  std::string mOutputFileName = "o2reco_t0.root";
  std::string mOutputTreeName = "o2sim";
  std::string mRPOutputBranchName = "T0Cluster";
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginT0;
};

/// create a processor spec
/// write ITS clusters a root file
framework::DataProcessorSpec getT0RecPointWriterSpec(bool useMC);

} // namespace t0
} // namespace o2

#endif /* O2_FIT_T0RECPOINTWRITER_H */
