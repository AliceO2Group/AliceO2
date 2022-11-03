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

/// \file   STFDecoderSpec.h
/// \brief  Device to decode ITS/MFT raw data from STF
/// \author ruben.shahoyan@cern.ch

#ifndef O2_ITSMFT_STFDECODER_
#define O2_ITSMFT_STFDECODER_

#include <TStopwatch.h>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <memory>
#include <string>
#include <string_view>
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "ITSMFTReconstruction/RawPixelDecoder.h"

using namespace o2::framework;

namespace o2
{
namespace base
{
class GRPGeomRequest;
}
namespace itsmft
{
class Clusterer;

struct STFDecoderInp {
  bool doClusters = true;
  bool doPatterns = true;
  bool doDigits = false;
  bool doCalib = false;
  bool doSquashing = false;
  bool askSTFDist = true;
  bool allowReporting = true;
  o2::header::DataOrigin origin{"NIL"};
  std::string deviceName{};
  std::string inputSpec{};
};

template <class Mapping>
class STFDecoder : public Task
{
 public:
  STFDecoder(const STFDecoderInp& inp, std::shared_ptr<o2::base::GRPGeomRequest> gr);
  STFDecoder() = default;
  ~STFDecoder() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final { finalize(); }
  void stop() final { finalize(); }
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  void finalize();
  std::unique_ptr<o2::itsmft::Clusterer> setupClusterer(const std::string& dictName);
  TStopwatch mTimer;
  bool mDoClusters = false;
  bool mDoPatterns = false;
  bool mDoDigits = false;
  bool mDoCalibData = false;
  bool mUnmutExtraLanes = false;
  bool mFinalizeDone = false;
  bool mAllowReporting = true;
  bool mApplyNoiseMap = true;
  bool mUseClusterDictionary = true;
  int mDumpOnError = 0;
  int mNThreads = 1;
  int mVerbosity = 0;
  size_t mTFCounter = 0;
  size_t mEstNDig = 0;
  size_t mEstNClus = 0;
  size_t mEstNClusPatt = 0;
  size_t mEstNCalib = 0;
  size_t mEstNROF = 0;
  std::string mInputSpec;
  std::string mSelfName;
  std::unique_ptr<RawPixelDecoder<Mapping>> mDecoder;
  std::unique_ptr<Clusterer> mClusterer;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
};

using STFDecoderITS = STFDecoder<ChipMappingITS>;
using STFDecoderMFT = STFDecoder<ChipMappingMFT>;

o2::framework::DataProcessorSpec getSTFDecoderSpec(const STFDecoderInp& inp);

} // namespace itsmft
} // namespace o2

#endif /* O2_ITS_DIGITREADER */
