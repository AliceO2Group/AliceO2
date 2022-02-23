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

/// @file   FilteredTFReaderSpec.h
/// @brief  Reader for the reconstructed and filtered TF

#ifndef O2_FILTERED_TF_READER_H
#define O2_FILTERED_TF_READER_H

#include <TFile.h>
#include <TTree.h>
#include <vector>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsGlobalTracking/FilteredRecoTF.h"

namespace o2::filtering
{

class FilteredTFReader : public o2::framework::Task
{

 public:
  FilteredTFReader(bool useMC = true);
  ~FilteredTFReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);

  o2::dataformats::FilteredRecoTF mFiltTF, *mFiltTFPtr = &mFiltTF;

  bool mUseMC = true; // use MC truth

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mInputFileName = "";
  std::string mInputTreeName = "o2sim";
  std::string mFTFBranchName = "FilteredRecoTF";
};

/// create a processor spec
/// read ITS track data from a root file
framework::DataProcessorSpec getFilteredTFReaderSpec(bool useMC = true);

} // namespace o2::filtering

#endif // O2_FILTERED_TF_READER_H
