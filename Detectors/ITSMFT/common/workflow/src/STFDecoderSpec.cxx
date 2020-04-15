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
/// \brief  Device to decode ITS raw data from STF
/// \author ruben.shahoyan@cern.ch

#include <vector>

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTReconstruction/RawPixelDecoder.h"
#include "ITSMFTReconstruction/Clusterer.h"
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
#include <string>

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
  mTimer.Stop();
  mTimer.Reset();
}

///_______________________________________
template <class Mapping>
STFDecoder<Mapping>::~STFDecoder()
{
  LOGF(INFO, "Total STF decoding%s timing (w/o disk IO): Cpu: %.3e Real: %.3e s in %d slots",
       mDoClusters ? "/clustering" : "", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter());
}

///_______________________________________
template <class Mapping>
void STFDecoder<Mapping>::init(InitContext& ic)
{
  LOG(INFO) << "STF decoder for " << Mapping::getName();
  mDecoder = std::make_unique<RawPixelDecoder<Mapping>>();
  mDecoder->init();

  auto detID = Mapping::getDetID();

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

    // settings for the fired pixel overflow masking
    const auto& alpParams = DPLAlpideParam<Mapping::getDetID()>::Instance();
    mClusterer->setMaxBCSeparationToMask(alpParams.roFrameLength / o2::constants::lhc::LHCBunchSpacingNS + 10);

    std::string dictFile = o2::base::NameConf::getDictionaryFileName(detID, mDictName, ".bin");
    if (o2::base::NameConf::pathExists(dictFile)) {
      mClusterer->loadDictionary(dictFile);
      LOG(INFO) << Mapping::getName() << " clusterer running with a provided dictionary: " << dictFile;
    } else {
      LOG(INFO) << "Dictionary " << dictFile << " is absent, " << Mapping::getName() << " clusterer expects cluster patterns";
    }
    mClusterer->print();
  }
}

template <class Mapping>
void STFDecoder<Mapping>::run(ProcessingContext& pc)
{
  int nSlots = pc.inputs().getNofParts(0);
  double timeCl = 0, timeCPU0 = mTimer.CpuTime(), timeReal0 = mTimer.RealTime();
  mTimer.Start(false);
  mDecoder->startNewTF(pc.inputs());
  auto orig = o2::header::gDataOriginITS;

  std::vector<Digit, boost::container::pmr::polymorphic_allocator<Digit>>* digVec = nullptr;
  std::vector<ROFRecord, boost::container::pmr::polymorphic_allocator<ROFRecord>>* digROFVec = nullptr;
  if (mDoDigits) {
    digVec = &pc.outputs().make<std::vector<Digit>>(Output{orig, "DIGITS", 0, Lifetime::Timeframe});
    digROFVec = &pc.outputs().make<std::vector<ROFRecord>>(Output{orig, "ITSDigitROF", 0, Lifetime::Timeframe});
  }
  using CLUSVECDUMMY = std::vector<Cluster, boost::container::pmr::polymorphic_allocator<Cluster>>;
  CLUSVECDUMMY* clusVecDUMMY = nullptr; // to pick the template!
  std::vector<CompClusterExt, boost::container::pmr::polymorphic_allocator<CompClusterExt>>* clusCompVec = nullptr;
  std::vector<ROFRecord, boost::container::pmr::polymorphic_allocator<ROFRecord>>* clusROFVec = nullptr;
  std::vector<unsigned char, boost::container::pmr::polymorphic_allocator<unsigned char>>* clusPattVec = nullptr;

  if (mDoClusters) { // we are not obliged to create vectors which are not requested, but other devices might not know the options of this one
    timeCl = mClusterer->getTimer().CpuTime();
    clusVecDUMMY = &pc.outputs().make<std::vector<o2::itsmft::Cluster>>(Output{orig, "CLUSTERS", 0, Lifetime::Timeframe});
    clusCompVec = &pc.outputs().make<std::vector<CompClusterExt>>(Output{orig, "COMPCLUSTERS", 0, Lifetime::Timeframe});
    clusPattVec = &pc.outputs().make<std::vector<unsigned char>>(Output{orig, "PATTERNS", 0, Lifetime::Timeframe});
    clusROFVec = &pc.outputs().make<std::vector<ROFRecord>>(Output{orig, "ITSClusterROF", 0, Lifetime::Timeframe});
  }

  // if digits are requested, we don't want clusterer to run automatic decoding
  mDecoder->setDecodeNextAuto(!(mDoDigits && mDoClusters));
  while (mDecoder->decodeNextTrigger()) {
    if (mDoDigits) {                                    // call before clusterization, since the latter will hide the digits
      mDecoder->fillDecodedDigits(*digVec, *digROFVec); // lot of copying involved
    }
    if (mDoClusters) {
      mClusterer->process(*mDecoder.get(), (CLUSVECDUMMY*)nullptr, clusCompVec, mDoPatterns ? clusPattVec : nullptr, clusROFVec);
    }
  }

  if (mDoClusters) {
    LOG(INFO) << "Built " << clusCompVec->size() << " clusters in " << clusROFVec->size() << " ROFs in "
              << mClusterer->getTimer().CpuTime() - timeCl << " s";
  }
  if (mDoDigits) {
    LOG(INFO) << "Decoded " << digVec->size() << " Digits in " << digROFVec->size() << " ROFs";
  }
  mTimer.Stop();
  LOG(INFO) << "Total time for this TF: CPU: " << mTimer.CpuTime() - timeCPU0 << " Real: " << mTimer.RealTime() - timeReal0;
}

///_______________________________________
template <class Mapping>
std::unique_ptr<Clusterer> STFDecoder<Mapping>::setupClusterer(const std::string& dictName)
{
  bool isContinuous = true;
  const auto grp = o2::parameters::GRPObject::loadFrom(o2::base::NameConf::getGRPFileName());
  if (grp) {
    isContinuous = grp->isDetContinuousReadOut(Mapping::getName());
  } else {
    throw std::runtime_error("failed to retrieve GRP");
  }
  // settings for the fired pixel overflow masking
  const auto& alpParams = DPLAlpideParam<Mapping::getDetID()>::Instance();
  int maskNBC = alpParams.roFrameLength / o2::constants::lhc::LHCBunchSpacingNS + 10;
  mClusterer = createClusterer(Mapping::getNChips(), maskNBC, isContinuous, "");
  mClusterer->print();
}

DataProcessorSpec getSTFDecoderITSSpec(bool doClusters, bool doPatterns, bool doDigits, const std::string& dict)
{
  std::vector<OutputSpec> outputs;
  auto orig = o2::header::gDataOriginITS;

  if (doDigits) {
    outputs.emplace_back(orig, "DIGITS", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "ITSDigitROF", 0, Lifetime::Timeframe);
  }
  if (doClusters) {
    outputs.emplace_back(orig, "COMPCLUSTERS", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "ITSClusterROF", 0, Lifetime::Timeframe);
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
    Options{}};
}

DataProcessorSpec getSTFDecoderMFTSpec(bool doClusters, bool doPatterns, bool doDigits, const std::string& dict)
{
  std::vector<OutputSpec> outputs;
  auto orig = o2::header::gDataOriginMFT;
  if (doDigits) {
    outputs.emplace_back(orig, "DIGMFT", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "MFTDigitROF", 0, Lifetime::Timeframe);
  }
  if (doClusters) {
    outputs.emplace_back(orig, "COMPCLUSTERS", 0, Lifetime::Timeframe);
    outputs.emplace_back(orig, "MFTClusterROF", 0, Lifetime::Timeframe);
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
    AlgorithmSpec{adaptFromTask<STFDecoder<ChipMappingITS>>(doClusters, doPatterns, doDigits, dict)},
    Options{}};
}

} // namespace itsmft
} // namespace o2
