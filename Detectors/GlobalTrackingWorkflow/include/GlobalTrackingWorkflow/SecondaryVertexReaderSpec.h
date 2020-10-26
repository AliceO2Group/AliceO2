// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   SecondaryVertexReaderSpec.h

#ifndef O2_SECONDARY_VERTEXREADER
#define O2_SECONDARY_VERTEXREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/V0.h"

namespace o2
{
namespace vertexing
{
// read secondary vertices produces by the o2-secondary-vertexing-workflow

class SecondaryVertexReader : public o2::framework::Task
{
  using RRef = o2::dataformats::RangeReference<int, int>;
  using V0 = o2::dataformats::V0;

 public:
  SecondaryVertexReader() = default;
  ~SecondaryVertexReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree();

  bool mVerbose = false;

  std::vector<V0> mV0s, *mV0sPtr = &mV0s;
  std::vector<RRef> mPV2V0Ref, *mPV2V0RefPtr = &mPV2V0Ref;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::string mFileNameMatches = "";
  std::string mSVertexTreeName = "o2sim";
  std::string mV0BranchName = "V0s";
  std::string mPVertex2V0RefBranchName = "PV2V0Refs";
};

/// create a processor spec
/// read secondary vertex data from a root file
o2::framework::DataProcessorSpec getSecondaryVertexReaderSpec();

} // namespace vertexing
} // namespace o2

#endif
