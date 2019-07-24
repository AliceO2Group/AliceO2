// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0RecPointWriterSpec.h

#ifndef O2_FT0RECPOINTWRITER_H
#define O2_FT0RECPOINTWRITER_H

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{

class FT0RecPointWriter : public Task
{
 public:
  FT0RecPointWriter(bool useMC = true) : mUseMC(useMC) {}
  ~FT0RecPointWriter() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  bool mFinished = false;
  bool mUseMC = true;

  std::string mOutputFileName = "o2reco_ft0.root";
  std::string mOutputTreeName = "o2sim";
  std::string mRPOutputBranchName = "FT0Cluster";
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginFT0;
};

/// create a processor spec
/// write ITS clusters a root file
framework::DataProcessorSpec getFT0RecPointWriterSpec(bool useMC);

} // namespace ft0
} // namespace o2

#endif /* O2_FT0RECPOINTWRITER_H */
