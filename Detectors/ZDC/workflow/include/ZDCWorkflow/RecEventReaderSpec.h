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

/// @file   RecEventReaderSpec.h

#ifndef O2_ZDC_RECEVENTREADER_SPEC
#define O2_ZDC_RECEVENTREADER_SPEC

#include "TFile.h"
#include "TTree.h"

#include "Framework/RootSerializationSupport.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsZDC/RecEvent.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

class RecEventReader : public Task
{
 public:
  RecEventReader(bool useMC = true);
  ~RecEventReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;

  bool mUseMC = true; // use MC truth
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginZDC;

  std::vector<o2::zdc::BCRecData>* mBCRecData = nullptr;
  std::vector<o2::zdc::ZDCEnergy>* mZDCEnergy = nullptr;
  std::vector<o2::zdc::ZDCTDCData>* mZDCTDCData = nullptr;
  std::vector<uint16_t>* mZDCInfo = nullptr;

  std::string mInputFileName = "zdcreco.root";
  std::string mRecEventTreeName = "o2rec";
  std::string mBCRecDataBranchName = "ZDCRecBC";
  std::string mZDCEnergyBranchName = "ZDCRecE";
  std::string mZDCTDCDataBranchName = "ZDCRecTDC";
  std::string mZDCInfoBranchName = "ZDCRecInfo";
};

/// create a processor spec
/// read reconstructed ZDC event parts
framework::DataProcessorSpec getRecEventReaderSpec(bool useMC);

} // namespace zdc
} // namespace o2

#endif /* O2_ZDC_RECEVENTREADER_SPEC */
