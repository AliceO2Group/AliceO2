// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   STFDecoderSpec.cxx
/// \brief  Device to decode ITS or MFT raw data from STF
/// \author ruben.shahoyan@cern.ch

#include <vector>

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTReconstruction/RawPixelDecoder.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "ITSMFTReconstruction/GBTLink.h"
#include "ITSMFTWorkflow/STFDecoderSpec.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/StringUtils.h"

namespace o2
{
namespace itsmft
{

using namespace o2::framework;

///_______________________________________
template <class Mapping>
STFDecoder<Mapping>::STFDecoder(bool doClusters, bool doPatterns, bool doDigits, bool doCalib, std::string_view dict, std::string_view noise)
  : mDoClusters(doClusters), mDoPatterns(doPatterns), mDoDigits(doDigits), mDoCalibData(doCalib), mDictName(dict), mNoiseName(noise)
{
  mSelfName = o2::utils::Str::concat_string(Mapping::getName(), "STFDecoder");
  mTimer.Stop();
  mTimer.Reset();
}

///_______________________________________
template <class Mapping>
void STFDecoder<Mapping>::init(InitContext& ic)
{
  mDecoder = std::make_unique<RawPixelDecoder<Mapping>>();
  mDecoder->init();

  auto detID = Mapping::getDetID();
  mNThreads = std::max(1, ic.options().get<int>("nthreads"));
  mDecoder->setNThreads(mNThreads);
  mDecoder->setFormat(ic.options().get<bool>("old-format") ? GBTLink::OldFormat : GBTLink::NewFormat);
  mDecoder->setVerbosity(ic.options().get<int>("decoder-verbosity"));
  mDecoder->setFillCalibData(mDoCalibData);
  std::string noiseFile = o2::base::NameConf::getAlpideClusterDictionaryFileName(detID, mNoiseName, ".root");
  if (o2::utils::Str::pathExists(noiseFile)) {
    TFile* f = TFile::Open(noiseFile.data(), "old");
    auto pnoise = (NoiseMap*)f->Get("Noise");
    AlpideCoder::setNoisyPixels(pnoise);
    LOG(INFO) << mSelfName << " loading noise map file: " << noiseFile;
  } else {
    LOG(INFO) << mSelfName << " Noise file " << noiseFile << " is absent, " << Mapping::getName() << " running without noise suppression";
  }

  if (mDoClusters) {
    mClusterer = std::make_unique<Clusterer>();
    mClusterer->setNChips(Mapping::getNChips());
    const auto grp = o2::parameters::GRPObject::loadFrom();
    if (grp) {
      mClusterer->setContinuousReadOut(grp->isDetContinuousReadOut(detID));
    } else {
      throw std::runtime_error("failed to retrieve GRP");
    }

    // settings for the fired pixel overflow masking
    const auto& alpParams = DPLAlpideParam<Mapping::getDetID()>::Instance();
    const auto& clParams = ClustererParam<Mapping::getDetID()>::Instance();
    auto nbc = clParams.maxBCDiffToMaskBias;
    nbc += mClusterer->isContinuousReadOut() ? alpParams.roFrameLengthInBC : (alpParams.roFrameLengthTrig / o2::constants::lhc::LHCBunchSpacingNS);
    mClusterer->setMaxBCSeparationToMask(nbc);
    mClusterer->setMaxRowColDiffToMask(clParams.maxRowColDiffToMask);

    std::string dictFile = o2::base::NameConf::getAlpideClusterDictionaryFileName(detID, mDictName, ".bin");
    if (o2::utils::Str::pathExists(dictFile)) {
      mClusterer->loadDictionary(dictFile);
      LOG(INFO) << mSelfName << " clusterer running with a provided dictionary: " << dictFile;
    } else {
      LOG(INFO) << mSelfName << " Dictionary " << dictFile << " is absent, " << Mapping::getName() << " clusterer expects cluster patterns";
    }
    mClusterer->print();
  }
}

///_______________________________________
template <class Mapping>
void STFDecoder<Mapping>::run(ProcessingContext& pc)
{
  int nSlots = pc.inputs().getNofParts(0);
  double timeCPU0 = mTimer.CpuTime(), timeReal0 = mTimer.RealTime();
  mTimer.Start(false);
  mDecoder->startNewTF(pc.inputs());
  auto orig = Mapping::getOrigin();
  std::vector<o2::itsmft::CompClusterExt> clusCompVec;
  std::vector<o2::itsmft::ROFRecord> clusROFVec;
  std::vector<unsigned char> clusPattVec;

  std::vector<Digit> digVec;
  std::vector<GBTCalibData> calVec;
  std::vector<ROFRecord> digROFVec;

  mDecoder->setDecodeNextAuto(false);
  while (mDecoder->decodeNextTrigger()) {
    if (mDoDigits) {                                    // call before clusterization, since the latter will hide the digits
      mDecoder->fillDecodedDigits(digVec, digROFVec);   // lot of copying involved
      if (mDoCalibData) {
        mDecoder->fillCalibData(calVec);
      }
    }

    if (mDoClusters) { // !!! THREADS !!!
      mClusterer->process(mNThreads, *mDecoder.get(), &clusCompVec, mDoPatterns ? &clusPattVec : nullptr, &clusROFVec);
    }
  }

  if (mDoDigits) {
    pc.outputs().snapshot(Output{orig, "DIGITS", 0, Lifetime::Timeframe}, digVec);
    pc.outputs().snapshot(Output{orig, "DIGITSROF", 0, Lifetime::Timeframe}, digROFVec);
    if (mDoCalibData) {
      pc.outputs().snapshot(Output{orig, "GBTCALIB", 0, Lifetime::Timeframe}, calVec);
    }
  }

  if (mDoClusters) {                                                                  // we are not obliged to create vectors which are not requested, but other devices might not know the options of this one
    pc.outputs().snapshot(Output{orig, "COMPCLUSTERS", 0, Lifetime::Timeframe}, clusCompVec);
    pc.outputs().snapshot(Output{orig, "PATTERNS", 0, Lifetime::Timeframe}, clusPattVec);
    pc.outputs().snapshot(Output{orig, "CLUSTERSROF", 0, Lifetime::Timeframe}, clusROFVec);
  }

  if (mDoClusters) {
    LOG(INFO) << mSelfName << " Built " << clusCompVec.size() << " clusters in " << clusROFVec.size() << " ROFs";
  }
  if (mDoDigits) {
    LOG(INFO) << mSelfName << " Decoded " << digVec.size() << " Digits in " << digROFVec.size() << " ROFs";
  }
  mTimer.Stop();
  auto tfID = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getByPos(0))->tfCounter;
  LOG(INFO) << mSelfName << " Total time for TF " << tfID << '(' << mTFCounter << ") : CPU: " << mTimer.CpuTime() - timeCPU0 << " Real: " << mTimer.RealTime() - timeReal0;
  mTFCounter++;
}

///_______________________________________
template <class Mapping>
void STFDecoder<Mapping>::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "%s statistics:", mSelfName);
  LOGF(INFO, "%s Total STF decoding%s timing (w/o disk IO): Cpu: %.3e Real: %.3e s in %d slots", mSelfName,
       mDoClusters ? "/clustering" : "", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
  if (mDecoder) {
    mDecoder->printReport();
  }
  if (mClusterer) {
    mClusterer->print();
  }
}

DataProcessorSpec getSTFDecoderITSSpec(bool doClusters, bool doPatterns, bool doDigits, bool doCalib, bool askDISTSTF, const std::string& dict, const std::string& noise)
{
  std::vector<OutputSpec> outputs;
  auto orig = o2::header::gDataOriginITS;

  if (doDigits) {
    outputs.emplace_back(orig, "DIGITS", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "DIGITSROF", 0, Lifetime::Timeframe);
    if (doCalib) {
      outputs.emplace_back(orig, "GBTCALIB", 0, Lifetime::Timeframe);
    }
  }
  if (doClusters) {
    outputs.emplace_back(orig, "COMPCLUSTERS", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "CLUSTERSROF", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "PATTERNS", 0, Lifetime::Timeframe);
  }

  std::vector<InputSpec> inputs{{"stf", ConcreteDataTypeMatcher{orig, "RAWDATA"}, Lifetime::Optional}};
  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-stf-decoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<STFDecoder<ChipMappingITS>>(doClusters, doPatterns, doDigits, doCalib, dict, noise)},
    Options{
      {"nthreads", VariantType::Int, 1, {"Number of decoding/clustering threads"}},
      {"old-format", VariantType::Bool, false, {"Use old format (1 trigger per CRU page)"}},
      {"decoder-verbosity", VariantType::Int, 0, {"Verbosity level (-1: silent, 0: errors, 1: headers, 2: data)"}}}};
}

DataProcessorSpec getSTFDecoderMFTSpec(bool doClusters, bool doPatterns, bool doDigits, bool doCalib, bool askDISTSTF, const std::string& dict, const std::string& noise)
{
  std::vector<OutputSpec> outputs;
  auto orig = o2::header::gDataOriginMFT;
  if (doDigits) {
    outputs.emplace_back(orig, "DIGITS", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "DIGITSROF", 0, Lifetime::Timeframe);
    if (doCalib) {
      outputs.emplace_back(orig, "GBTCALIB", 0, Lifetime::Timeframe);
    }
  }
  if (doClusters) {
    outputs.emplace_back(orig, "COMPCLUSTERS", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "CLUSTERSROF", 0, Lifetime::Timeframe);
    // in principle, we don't need to open this input if we don't need to send real data,
    // but other devices expecting it do not know about options of this device: problem?
    // if (doClusters && doPatterns)
    outputs.emplace_back(orig, "PATTERNS", 0, Lifetime::Timeframe);
  }

  std::vector<InputSpec> inputs{{"stf", ConcreteDataTypeMatcher{orig, "RAWDATA"}, Lifetime::Optional}};
  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "mft-stf-decoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<STFDecoder<ChipMappingMFT>>(doClusters, doPatterns, doDigits, doCalib, dict, noise)},
    Options{
      {"nthreads", VariantType::Int, 1, {"Number of decoding/clustering threads"}},
      {"old-format", VariantType::Bool, false, {"Use old format (1 trigger per CRU page)"}},
      {"decoder-verbosity", VariantType::Int, 0, {"Verbosity level (-1: silent, 0: errors, 1: headers, 2: data)"}}}};
}

} // namespace itsmft
} // namespace o2
