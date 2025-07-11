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

#ifndef O2_ITSMFT_DIGITREADER
#define O2_ITSMFT_DIGITREADER

#include "TFile.h"
#include "TTree.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/GBTCalibData.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace itsmft
{

template <int N>
class DigitReader : public Task
{
 public:
  static constexpr o2::detectors::DetID ID{N == o2::detectors::DetID::ITS ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT};
  static constexpr o2::header::DataOrigin Origin{N == o2::detectors::DetID::ITS ? o2::header::gDataOriginITS : o2::header::gDataOriginMFT};
  static constexpr int NLayers{o2::itsmft::DPLAlpideParam<N>::getNLayers()};
  static constexpr int RLayers = o2::itsmft::DPLAlpideParam<N>::supportsStaggering() ? NLayers : 1;

  DigitReader() = delete;
  DigitReader(bool useMC, bool useCalib, bool triggerOut);
  ~DigitReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);
  template <typename Ptr>
  void setBranchAddress(const std::string& base, Ptr& addr, int layer = -1);
  std::string getBranchName(const std::string& base, int index);

  std::array<std::vector<o2::itsmft::Digit>*, NLayers> mDigits;
  std::vector<o2::itsmft::GBTCalibData> mCalib, *mCalibPtr = &mCalib;
  std::array<std::vector<o2::itsmft::ROFRecord>*, NLayers> mDigROFRec;
  std::array<std::vector<o2::itsmft::MC2ROFRecord>*, NLayers> mDigMC2ROFs;
  std::array<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>, NLayers> mConstLabels;
  std::array<o2::dataformats::IOMCTruthContainerView*, NLayers> mPLabels;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  bool mUseMC = true;        // use MC truth
  bool mUseCalib = true;     // send calib data
  bool mTriggerOut = true;   // send dummy triggers vector
  bool mUseIRFrames = false; // selected IRFrames modes
  int mROFBiasInBC = 0;
  int mROFLengthInBC = 0;
  int mNRUs = 0;
  std::string mDetName;
  std::string mDetNameLC;
  std::string mFileName;
  std::string mDigTreeName = "o2sim";
  std::string mDigitBranchName = "Digit";
  std::string mDigitROFBranchName = "DigitROF";
  std::string mCalibBranchName = "Calib";

  std::string mDigitMCTruthBranchName = "DigitMCTruth";
  std::string mDigitMC2ROFBranchName = "DigitMC2ROF";
};

class ITSDigitReader : public DigitReader<o2::detectors::DetID::ITS>
{
 public:
  ITSDigitReader(bool useMC = true, bool useCalib = false, bool useTriggers = true)
    : DigitReader<o2::detectors::DetID::ITS>(useMC, useCalib, useTriggers) {}
};

class MFTDigitReader : public DigitReader<o2::detectors::DetID::MFT>
{
 public:
  MFTDigitReader(bool useMC = true, bool useCalib = false, bool useTriggers = true)
    : DigitReader<o2::detectors::DetID::MFT>(useMC, useCalib, useTriggers) {}
};

/// create a processor spec
/// read ITS/MFT Digit data from a root file
framework::DataProcessorSpec getITSDigitReaderSpec(bool useMC = true, bool useCalib = false, bool useTriggers = true, std::string defname = "o2_itsdigits.root");
framework::DataProcessorSpec getMFTDigitReaderSpec(bool useMC = true, bool useCalib = false, bool useTriggers = true, std::string defname = "o2_mftdigits.root");

} // namespace itsmft
} // namespace o2

#endif /* O2_ITSMFT_DigitREADER */
