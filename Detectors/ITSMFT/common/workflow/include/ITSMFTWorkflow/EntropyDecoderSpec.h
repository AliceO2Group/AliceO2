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

/// @file   EntropyDecoderSpec.h
/// @brief  Convert CTF (EncodedBlocks) to clusters streams

#ifndef O2_ITSMFT_ENTROPYDECODER_SPEC
#define O2_ITSMFT_ENTROPYDECODER_SPEC

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "ITSMFTReconstruction/CTFCoder.h"
#include "DataFormatsITSMFT/NoiseMap.h"
#include "ITSMFTReconstruction/LookUp.h"
#include <TStopwatch.h>
#include <memory>

namespace o2
{
namespace itsmft
{

template <int N>
class EntropyDecoderSpec : public o2::framework::Task
{
 public:
  EntropyDecoderSpec(int verbosity, bool doStag, bool getDigits = false, const std::string& ctfdictOpt = "none");
  ~EntropyDecoderSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

  static std::string getBinding(const std::string& name, int spec);
  static constexpr o2::detectors::DetID ID{N == o2::detectors::DetID::ITS ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT};
  static constexpr o2::header::DataOrigin Origin{N == o2::detectors::DetID::ITS ? o2::header::gDataOriginITS : o2::header::gDataOriginMFT};

 private:
  void updateTimeDependentParams(o2::framework::ProcessingContext& pc);

  o2::itsmft::CTFCoder<N> mCTFCoder;
  const NoiseMap* mNoiseMap = nullptr;
  LookUp mPattIdConverter;
  bool mDoStaggering{false};
  bool mGetDigits{false};
  bool mMaskNoise{false};
  bool mUseClusterDictionary{true};

  std::string mCTFDictPath{};
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getITSEntropyDecoderSpec(int verbosity, bool doStag, bool getDigits, unsigned int sspec, const std::string& ctfdictOpt);
framework::DataProcessorSpec getMFTEntropyDecoderSpec(int verbosity, bool doStag, bool getDigits, unsigned int sspec, const std::string& ctfdictOpt);

} // namespace itsmft
} // namespace o2

#endif
