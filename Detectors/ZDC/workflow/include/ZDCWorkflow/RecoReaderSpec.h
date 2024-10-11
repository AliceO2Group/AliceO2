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

#ifndef O2_ZDC_RECOREADERSPEC_H
#define O2_ZDC_RECOREADERSPEC_H

#include "TFile.h"
#include "TTree.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace zdc
{

class RecoReader : public framework::Task
{
 public:
  RecoReader() = default;
  ~RecoReader() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;

 private:
  std::unique_ptr<TTree> mTree;
  std::unique_ptr<TFile> mFile;
};

/// create a processor spec
/// read reconstructed ZDC data from a root file
framework::DataProcessorSpec getRecoReaderSpec();

} // end namespace zdc
} // end namespace o2

#endif // O2_ZDC_RECOREADERSPEC_H
