// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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

#include <memory>
#include <string>
#include <vector>
#include <TStopwatch.h>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "DataFormatsITSMFT/ROFRecord.h"
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
  bool doStaggering = false;
  bool askSTFDist = true;
  bool allowReporting = true;
  bool verifyDecoder = false;
  o2::header::DataOrigin origin{"NIL"};
  std::string deviceName{};
  std::string inputSpec{};
};

template <class Mapping>
class STFDecoder : public Task
{
  using AlpideParam = DPLAlpideParam<Mapping::getDetID()>;

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
  void reset();
  std::unique_ptr<o2::itsmft::Clusterer> setupClusterer(const std::string& dictName);
  bool ensureContinuousROF(const std::vector<ROFRecord>& in, std::vector<ROFRecord>& out, int lr, int nROFsTF, const char* name);
  void rectifyDigits(std::vector<ROFRecord>& rofVec, std::vector<Digit>& digVec);
  void rectifyClusters(std::vector<ROFRecord>& rofVec, std::vector<CompClusterExt>& clusVec, std::vector<unsigned char>& pattVec);

  TStopwatch mTimer;
  bool mDoClusters = false;
  bool mDoPatterns = false;
  bool mDoDigits = false;
  bool mDoCalibData = false;
  bool mDoStaggering = false;
  bool mUnmutExtraLanes = false;
  bool mFinalizeDone = false;
  bool mAllowReporting = true;
  bool mApplyNoiseMap = true;
  bool mUseClusterDictionary = true;
  bool mVerifyDecoder = false;
  bool mDumpFrom1stPipeline = false;
  int mDumpOnError = 0;
  int mNThreads = 1;
  int mVerbosity = 0;
  int mLayers = 1;
  long mROFErrRepIntervalMS = 0;
  size_t mTFCounter = 0;
  uint32_t mFirstTFOrbit = 0;
  o2::InteractionRecord mFirstIR;
  std::vector<size_t> mEstNDig{0};
  std::vector<size_t> mEstNClus{0};
  std::vector<size_t> mEstNClusPatt{0};
  std::vector<size_t> mEstNCalib{0};
  size_t mMaxRawDumpsSize = 0;
  size_t mRawDumpedSize = 0;
  std::string mInputSpec;
  std::string mSelfName;
  std::vector<std::unique_ptr<RawPixelDecoder<Mapping>>> mDecoder;
  std::unique_ptr<Clusterer> mClusterer;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
};

using STFDecoderITS = STFDecoder<ChipMappingITS>;
using STFDecoderMFT = STFDecoder<ChipMappingMFT>;

o2::framework::DataProcessorSpec getSTFDecoderSpec(const STFDecoderInp& inp);

} // namespace itsmft
} // namespace o2

#endif /* O2_ITS_DIGITREADER */
