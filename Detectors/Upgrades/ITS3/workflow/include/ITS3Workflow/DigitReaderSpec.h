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

/// @file   DigitReaderSpec.h

#ifndef O2_ITS3_DIGITREADER
#define O2_ITS3_DIGITREADER

#include <array>

#include <TFile.h>
#include <TTree.h>

#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"

using namespace o2::framework;

namespace o2::its3
{

class ITS3DigitReader final : public Task
{
 public:
  static constexpr int NLayers = 7;

  ITS3DigitReader(bool useMC, bool doStag, bool useCalib);
  ~ITS3DigitReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);

  std::array<std::vector<o2::itsmft::Digit>*, NLayers> mDigits{};
  std::array<std::vector<o2::itsmft::ROFRecord>*, NLayers> mDigROFRec{};
  std::array<std::vector<o2::itsmft::MC2ROFRecord>*, NLayers> mDigMC2ROFs{};
  std::vector<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>> mConstLabels;
  std::array<o2::dataformats::IOMCTruthContainerView*, NLayers> mPLabels{};

  const o2::header::DataOrigin mOrigin = o2::header::gDataOriginIT3;

  std::string getBranchName(const std::string& base, int index) const
  {
    if (mDoStaggering) {
      return base + "_" + std::to_string(index);
    }
    return base;
  }

  template <typename Ptr>
  void setBranchAddress(const std::string& base, Ptr& addr, int layer);

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;

  bool mUseMC = true;    // use MC truth
  bool mUseCalib = true; // send calib data
  bool mDoStaggering = false;

  std::string mDetName;
  std::string mDetNameLC;
  std::string mFileName;
  std::string mDigTreeName = "o2sim";
  std::string mDigBranchName = "Digit";
  std::string mDigROFBranchName = "DigitROF";
  std::string mDigMCTruthBranchName = "DigitMCTruth";
};

/// create a processor spec
/// read ITS/MFT Digit data from a root file
framework::DataProcessorSpec getITS3DigitReaderSpec(bool useMC = true, bool doStag = false, bool useCalib = false, std::string defname = "it3digits.root");

} // namespace o2::its3

#endif /* O2_ITS3_DigitREADER */
