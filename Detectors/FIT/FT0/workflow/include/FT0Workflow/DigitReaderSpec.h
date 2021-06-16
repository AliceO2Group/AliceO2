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

#ifndef O2_FT0_DIGITREADERSPEC_H
#define O2_FT0_DIGITREADERSPEC_H

#include "TFile.h"
#include "TTree.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace ft0
{

class DigitReader : public framework::Task
{
 public:
  DigitReader(bool useMC, bool useTrgInput) : mUseMC(useMC), mUseTrgInput(useTrgInput) {}
  ~DigitReader() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;

 private:
  bool mUseMC = true;
  bool mUseTrgInput = true;
  std::unique_ptr<TTree> mTree;
  std::unique_ptr<TFile> mFile;
};

/// create a processor spec
/// read simulated FT0 digits from a root file
framework::DataProcessorSpec getDigitReaderSpec(bool useMC, bool useTrgInput = true);

} // end namespace ft0
} // end namespace o2

#endif // O2_FT0_DIGITREADERSPEC_H
