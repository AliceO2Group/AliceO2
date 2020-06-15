// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitWriterSpec.h

#ifndef O2_MFT_DIGITWRITER_H_
#define O2_MFT_DIGITWRITER_H_

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace mft
{

class DigitWriter : public o2::framework::Task
{
 public:
  DigitWriter() = default;
  ~DigitWriter() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  std::unique_ptr<TFile> mFile = nullptr;
  std::unique_ptr<TTree> mTree = nullptr;
};

/// create a processor spec
/// write MFT digits a root file
framework::DataProcessorSpec getDigitWriterSpec();

} // namespace mft
} // namespace o2

#endif /* O2_MFT_DIGITWRITER_H */
