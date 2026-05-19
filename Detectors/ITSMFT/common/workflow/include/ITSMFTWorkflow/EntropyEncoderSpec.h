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

/// @file   EntropyEncoderSpec.h
/// @brief  Convert clusters streams to CTF (EncodedBlocks)

#ifndef O2_ITSMFT_ENTROPYENCODER_SPEC
#define O2_ITSMFT_ENTROPYENCODER_SPEC

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include <TStopwatch.h>
#include "ITSMFTReconstruction/CTFCoder.h"
#include "ITSMFTReconstruction/LookUp.h"

namespace o2
{
namespace itsmft
{

template <int N>
class EntropyEncoderSpec : public o2::framework::Task
{

 public:
  EntropyEncoderSpec(bool doStag, bool selIR, const std::string& ctfdictOpt = "none");
  ~EntropyEncoderSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void updateTimeDependentParams(o2::framework::ProcessingContext& pc);
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

  static std::string getBinding(const std::string& name, int spec);
  static constexpr o2::detectors::DetID ID{N == o2::detectors::DetID::ITS ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT};
  static constexpr o2::header::DataOrigin Origin{N == o2::detectors::DetID::ITS ? o2::header::gDataOriginITS : o2::header::gDataOriginMFT};

 private:
  o2::itsmft::CTFCoder<N> mCTFCoder;
  LookUp mPattIdConverter;
  bool mSelIR = false;
  bool mDoStaggering = false;
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getITSEntropyEncoderSpec(bool doStag = false, bool selIR = false, const std::string& ctfdictOpt = "none");
framework::DataProcessorSpec getMFTEntropyEncoderSpec(bool doStag = false, bool selIR = false, const std::string& ctfdictOpt = "none");

} // namespace itsmft
} // namespace o2

#endif
