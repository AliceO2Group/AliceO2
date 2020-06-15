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
#include "ITSBase/GeometryTGeo.h"
#include "MFTBase/GeometryTGeo.h"
#include "ITSMFTBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/StringUtils.h"

namespace o2
{
namespace itsmft
{

using namespace o2::framework;

///_______________________________________
template <class Mapping>
STFDecoder<Mapping>::STFDecoder(bool doClusters, bool doPatterns, bool doDigits, std::string_view dict)
  : mDoClusters(doClusters), mDoPatterns(doPatterns), mDoDigits(doDigits), mDictName(dict)
{
  mSelfName = o2::utils::concat_string(Mapping::getName(), "STFDecoder");
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
  mNThreads = ic.options().get<int>("nthreads");
  mDecoder->setNThreads(mNThreads);
  mDecoder->setFormat(ic.options().get<bool>("old-format") ? GBTLink::OldFormat : GBTLink::NewFormat);
  mDecoder->setVerbosity(ic.options().get<int>("decoder-verbosity"));
  if (mDoClusters) {
    o2::base::GeometryManager::loadGeometry(); // for generating full clusters
    GeometryTGeo* geom = nullptr;
    if (detID == o2::detectors::DetID::ITS) {
      geom = o2::its::GeometryTGeo::Instance();
      geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L));
    } else {
      geom = o2::mft::GeometryTGeo::Instance();
      geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L));
    }
    mClusterer = std::make_unique<Clusterer>();
    mClusterer->setGeometry(geom);
    mClusterer->setNChips(Mapping::getNChips());
    const auto grp = o2::parameters::GRPObject::loadFrom(o2::base::NameConf::getGRPFileName());
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

    std::string dictFile = o2::base::NameConf::getDictionaryFileName(detID, mDictName, ".bin");
    if (o2::base::NameConf::pathExists(dictFile)) {
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
  using CLUSVECDUMMY = std::vector<Cluster>;
  std::vector<o2::itsmft::Cluster> clusVec;
  std::vector<o2::itsmft::CompClusterExt> clusCompVec;
  std::vector<o2::itsmft::ROFRecord> clusROFVec;
  std::vector<unsigned char> clusPattVec;
  std::vector<Digit> digVec;
  std::vector<ROFRecord> digROFVec;
  CLUSVECDUMMY* clusVecDUMMY = nullptr;
  mDecoder->setDecodeNextAuto(false);
  while (mDecoder->decodeNextTrigger()) {
    if (mDoDigits) {                                    // call before clusterization, since the latter will hide the digits
      mDecoder->fillDecodedDigits(digVec, digROFVec);   // lot of copying involved
    }
    if (mDoClusters) { // !!! THREADS !!!
      mClusterer->process(mNThreads, *mDecoder.get(), (CLUSVECDUMMY*)nullptr, &clusCompVec, mDoPatterns ? &clusPattVec : nullptr, &clusROFVec);
    }
  }

  if (mDoDigits) {
    pc.outputs().snapshot(Output{orig, "DIGITS", 0, Lifetime::Timeframe}, digVec);
    pc.outputs().snapshot(Output{orig, "DigitROF", 0, Lifetime::Timeframe}, digROFVec);
  }
  if (mDoClusters) {                                                                  // we are not obliged to create vectors which are not requested, but other devices might not know the options of this one
    pc.outputs().snapshot(Output{orig, "CLUSTERS", 0, Lifetime::Timeframe}, clusVec); // DUMMY!!!
    pc.outputs().snapshot(Output{orig, "COMPCLUSTERS", 0, Lifetime::Timeframe}, clusCompVec);
    pc.outputs().snapshot(Output{orig, "PATTERNS", 0, Lifetime::Timeframe}, clusPattVec);
    pc.outputs().snapshot(Output{orig, "ClusterROF", 0, Lifetime::Timeframe}, clusROFVec);
  }

  if (mDoClusters) {
    LOG(INFO) << mSelfName << " Built " << clusCompVec.size() << " clusters in " << clusROFVec.size() << " ROFs";
  }
  if (mDoDigits) {
    LOG(INFO) << mSelfName << " Decoded " << digVec.size() << " Digits in " << digROFVec.size() << " ROFs";
  }
  mTimer.Stop();
  LOG(INFO) << mSelfName << " Total time for TF " << mTFCounter << " : CPU: " << mTimer.CpuTime() - timeCPU0 << " Real: " << mTimer.RealTime() - timeReal0;
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

DataProcessorSpec getSTFDecoderITSSpec(bool doClusters, bool doPatterns, bool doDigits, const std::string& dict)
{
  std::vector<OutputSpec> outputs;
  auto orig = o2::header::gDataOriginITS;

  if (doDigits) {
    outputs.emplace_back(orig, "DIGITS", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "DigitROF", 0, Lifetime::Timeframe);
  }
  if (doClusters) {
    outputs.emplace_back(orig, "COMPCLUSTERS", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "ClusterROF", 0, Lifetime::Timeframe);
    // in principle, we don't need to open this input if we don't need to send real data,
    // but other devices expecting it do not know about options of this device: problem?
    // if (doClusters && doPatterns)
    outputs.emplace_back(orig, "PATTERNS", 0, Lifetime::Timeframe); // RSTODO: DUMMY, FULL CLUSTERS ARE BEING ELIMINATED
    //
    outputs.emplace_back(orig, "CLUSTERS", 0, Lifetime::Timeframe); // RSTODO: DUMMY, FULL CLUSTERS ARE BEING ELIMINATED
  }

  return DataProcessorSpec{
    "its-stf-decoder",
    Inputs{{"stf", ConcreteDataTypeMatcher{orig, "RAWDATA"}, Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<STFDecoder<ChipMappingITS>>(doClusters, doPatterns, doDigits, dict)},
    Options{
      {"nthreads", VariantType::Int, 0, {"Number of decoding/clustering threads (<1: rely on openMP default)"}},
      {"old-format", VariantType::Bool, false, {"Use old format (1 trigger per CRU page)"}},
      {"decoder-verbosity", VariantType::Int, 0, {"Verbosity level (-1: silent, 0: errors, 1: headers, 2: data)"}}}};
}

DataProcessorSpec getSTFDecoderMFTSpec(bool doClusters, bool doPatterns, bool doDigits, const std::string& dict)
{
  std::vector<OutputSpec> outputs;
  auto orig = o2::header::gDataOriginMFT;
  if (doDigits) {
    outputs.emplace_back(orig, "DIGITS", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "DigitROF", 0, Lifetime::Timeframe);
  }
  if (doClusters) {
    outputs.emplace_back(orig, "COMPCLUSTERS", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "ClusterROF", 0, Lifetime::Timeframe);
    // in principle, we don't need to open this input if we don't need to send real data,
    // but other devices expecting it do not know about options of this device: problem?
    // if (doClusters && doPatterns)
    outputs.emplace_back(orig, "PATTERNS", 0, Lifetime::Timeframe);
    //
    outputs.emplace_back(orig, "CLUSTERS", 0, Lifetime::Timeframe); // RSTODO: DUMMY, FULL CLUSTERS ARE BEING ELIMINATED
  }

  return DataProcessorSpec{
    "mft-stf-decoder",
    Inputs{{"stf", ConcreteDataTypeMatcher{orig, "RAWDATA"}, Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<STFDecoder<ChipMappingMFT>>(doClusters, doPatterns, doDigits, dict)},
    Options{
      {"nthreads", VariantType::Int, 0, {"Number of decoding/clustering threads (<1: rely on openMP default)"}},
      {"old-format", VariantType::Bool, false, {"Use old format (1 trigger per CRU page)"}},
      {"decoder-verbosity", VariantType::Int, 0, {"Verbosity level (-1: silent, 0: errors, 1: headers, 2: data)"}}}};
}

} // namespace itsmft
} // namespace o2
