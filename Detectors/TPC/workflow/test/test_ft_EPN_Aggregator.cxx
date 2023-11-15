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

/// \file  test_ft_EPN_Aggregator.cxx
/// \brief this task tests the the calculation of fourier coefficients on EPN and aggregator
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#include "TPCWorkflow/TPCFLPIDCSpec.h"
#include "TPCWorkflow/TPCIntegrateIDCSpec.h"
#include "TPCWorkflow/TPCFourierTransformEPNSpec.h"
#include "TPCWorkflow/TPCFourierTransformAggregatorSpec.h"
#include "TPCWorkflow/TPCDistributeIDCSpec.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DetectorsRaw/HBFUtils.h"
#include "Framework/CompletionPolicyHelpers.h"

#include <boost/throw_exception.hpp>
#include "Framework/DataRefUtils.h"
#include "Framework/ControlService.h"
#include "Algorithm/RangeTokenizer.h"
#include "TRandom.h"
#include "TKey.h"

#include <unistd.h>

using namespace o2::framework;

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(fatal) << R"(Test condition ")" #condition R"(" failed)"; \
  }

DataProcessorSpec generateIDCsCRU(int lane, const unsigned int maxTFs, const std::vector<uint32_t>& crus, const bool delay, const bool loadFromFile, const int dropTFsRandom, const std::vector<int>& rangeTFsDrop);
DataProcessorSpec receiveFourierCoeffEPN(const unsigned int maxTFs, const unsigned int nFourierCoefficients);
DataProcessorSpec compare_EPN_AGG();
static constexpr o2::header::DataDescription getDataDescriptionCoeffEPN() { return o2::header::DataDescription{"COEFFEPNALL"}; }

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  const std::string cruDefault = "0-" + std::to_string(o2::tpc::CRU::MaxCRU - 1);

  std::vector<ConfigParamSpec> options{
    {"crus", VariantType::String, cruDefault.c_str(), {"List of CRUs, comma separated ranges, e.g. 0-3,7,9-15"}},
    {"timeframes", VariantType::Int, 20, {"Number of TFs which will be produced"}},
    {"ion-drift-time", VariantType::Int, 50, {"Ion drift time in ms. (Number of 1D-IDCs which will be used for the calculation of the fourier coefficients.)"}},
    {"nFourierCoeff", VariantType::Int, 20, {"Number of fourier coefficients per TF (real+imag) which will be compared. The maximum can be 'ion-drift-time + 2'."}},
    {"use-naive-fft", VariantType::Bool, false, {"using naive fourier transform (true) or FFTW (false)"}},
    {"seed", VariantType::Int, 0, {"Seed for the random IDC generator."}},
    {"only-idc-gen", VariantType::Bool, false, {"Start only the IDC generator device"}},
    {"load-from-file", VariantType::Bool, false, {"load from file"}},
    {"debug", VariantType::Bool, false, {"create debug for FT"}},
    {"idc-gen-lanes", VariantType::Int, 1, {"number of parallel lanes for generation of IDCs"}},
    {"idc-gen-time-lanes", VariantType::Int, 1, {"number of parallel lanes for generation of IDCs"}},
    {"delay", VariantType::Bool, false, {"Add delay for sending IDCs"}},
    {"dropTFsRandom", VariantType::Int, 0, {"Drop randomly whole TFs every dropTFsRandom TFs (for all CRUs)"}},
    {"dropTFsRange", VariantType::String, "", {"Drop range of TFs"}},
    {"hbfutils-config", VariantType::String, "hbfutils", {"config file for HBFUtils (or none) to get number of orbits per TF"}},
    {"nthreads", VariantType::Int, 1, {"Number of threads."}},
    {"iter", VariantType::Int, 0, {"Iteration for testing the workflow (.....)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  policies.push_back(CompletionPolicyHelpers::defineByName("tpc-factorize-idc.*", CompletionPolicy::CompletionOp::Consume));
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  const auto tpcCRUs = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("crus"));
  const auto rangeTFsDrop = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("dropTFsRange"));
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  const auto nCRUs = tpcCRUs.size();
  const auto first = tpcCRUs.begin();
  const auto last = std::min(tpcCRUs.end(), first + nCRUs);
  const std::vector<uint32_t> crus(first, last);
  const auto timeframes = static_cast<unsigned int>(config.options().get<int>("timeframes"));
  const auto iondrifttime = static_cast<unsigned int>(config.options().get<int>("ion-drift-time"));
  const auto nFourierCoefficients = std::clamp(static_cast<unsigned int>(config.options().get<int>("nFourierCoeff")), static_cast<unsigned int>(0), iondrifttime + 2);
  const auto nthreads = static_cast<unsigned int>(config.options().get<int>("nthreads"));
  const auto iter = static_cast<unsigned int>(config.options().get<int>("iter"));
  const auto seed = static_cast<unsigned int>(config.options().get<int>("seed"));
  const auto idcgenlanes = static_cast<unsigned int>(config.options().get<int>("idc-gen-lanes"));
  const auto idcgentimelanes = static_cast<unsigned int>(config.options().get<int>("idc-gen-time-lanes"));
  const auto delay = static_cast<unsigned int>(config.options().get<bool>("delay"));

  const bool fft = config.options().get<bool>("use-naive-fft");
  const bool onlyIDCGen = config.options().get<bool>("only-idc-gen");
  const bool debugFT = config.options().get<bool>("debug");
  const int dropTFsRandom = config.options().get<int>("dropTFsRandom");

  const unsigned int firstTF = 0;
  const unsigned int nLanes = 1;
  const bool loadFromFileGen = config.options().get<bool>("load-from-file");
  gRandom->SetSeed(seed);

  WorkflowSpec workflow;
  for (int ilane = 0; ilane < idcgenlanes; ++ilane) {
    const auto crusPerLane = nCRUs / idcgenlanes + ((nCRUs % idcgenlanes) != 0);
    const auto first = tpcCRUs.begin() + ilane * crusPerLane;
    if (first >= tpcCRUs.end()) {
      break;
    }
    const auto last = std::min(tpcCRUs.end(), first + crusPerLane);
    const std::vector<uint32_t> rangeCRUs(first, last);
    workflow.emplace_back(timePipeline(generateIDCsCRU(ilane, timeframes, rangeCRUs, delay, loadFromFileGen, dropTFsRandom, rangeTFsDrop), idcgentimelanes));
  }

  auto& hbfu = o2::raw::HBFUtils::Instance();
  long startTime = hbfu.startTime > 0 ? hbfu.startTime : std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  o2::conf::ConfigurableParam::updateFromString(fmt::format("HBFUtils.startTime={}", startTime).data());
  o2::raw::HBFUtilsInitializer hbfIni(config, workflow);

  if (onlyIDCGen == true) {
    return workflow;
  }

  if (iter == 0) {
    auto workflowTmp = WorkflowSpec{getTPCFLPIDCSpec(0, crus, iondrifttime, false, "", true, false), getTPCDistributeIDCSpec(0, crus, timeframes, nLanes, firstTF, false), getTPCFactorizeIDCSpec(0, crus, timeframes, timeframes, o2::tpc::IDCDeltaCompression::NO, false, true, false)};
    for (auto& spec : workflowTmp) {
      workflow.emplace_back(spec);
    }
  } else if (iter == 1) {
    const Side side = CRU(crus.front()).side();
    const std::string idc0File = (side == Side::A) ? fmt::format("IDCZero_A_{:02}.root", timeframes - 1) : fmt::format("IDCZero_C_{:02}.root", timeframes - 1);
    auto workflowTmp = WorkflowSpec{getTPCFLPIDCSpec(0, crus, iondrifttime, false, idc0File.data(), true, true), getTPCFourierTransformEPNSpec(crus, iondrifttime, nFourierCoefficients), getTPCDistributeIDCSpec(0, crus, timeframes, nLanes, firstTF, false), getTPCFactorizeIDCSpec(0, crus, timeframes, timeframes, o2::tpc::IDCDeltaCompression::NO, false, true, false), getTPCFourierTransformAggregatorSpec(iondrifttime, nFourierCoefficients, true, false, nLanes), receiveFourierCoeffEPN(timeframes, nFourierCoefficients), compare_EPN_AGG()};
    for (auto& spec : workflowTmp) {
      workflow.emplace_back(spec);
    }
  }

  IDCAverageGroup<IDCAverageGroupTPC>::setNThreads(nthreads);
  TPCFourierTransformEPNSpec::IDCFType::setFFT(!fft);
  TPCFourierTransformAggregatorSpec::IDCFType::setNThreads(nthreads);
  TPCFourierTransformAggregatorSpec::IDCFType::setFFT(!fft);
  return workflow;
}

DataProcessorSpec generateIDCsCRU(int lane, const unsigned int maxTFs, const std::vector<uint32_t>& crus, const bool delay, const bool loadFromFile, const int dropTFsRandom, const std::vector<int>& rangeTFsDrop)
{
  using timer = std::chrono::high_resolution_clock;

  std::array<std::vector<float>, CRU::MaxCRU> mIDCs{};
  if (loadFromFile) {
    TFile fInp("IDCGroup.root", "READ");
    for (TObject* keyAsObj : *fInp.GetListOfKeys()) {
      const auto key = dynamic_cast<TKey*>(keyAsObj);
      LOGP(info, "Key name: {} Type: {}", key->GetName(), key->GetClassName());
      std::vector<float>* idcData = (std::vector<float>*)fInp.Get(key->GetName());
      std::string name = key->GetName();
      const auto pos = name.find_last_of('_') + 1;
      const int cru = std::stoi(name.substr(pos, name.length()));
      mIDCs[cru] = *idcData;
      LOGP(info, "Loaded {} IDCs from file for CRU {}", mIDCs[cru].size(), cru);
      delete idcData;
    }
  }

  // generate random IDCs per CRU
  const int nOrbitsPerTF = 128; // number of orbits per TF
  const int nOrbitsPerIDC = 12; // integration interval in units of orbits per IDC
  const int nIDCs = nOrbitsPerTF / nOrbitsPerIDC;

  std::vector<OutputSpec> outputSpecs;
  outputSpecs.reserve(crus.size());
  for (const auto& cru : crus) {
    const o2::header::DataHeader::SubSpecificationType subSpec{cru << 7};
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCIntegrateIDCDevice::getDataDescription(TPCIntegrateIDCDevice::IDCFormat::Sim), subSpec});
  }

  gRandom->SetSeed(42);
  return DataProcessorSpec{
    fmt::format("idc-generator-{:02}", lane).data(),
    Inputs{},
    outputSpecs,
    AlgorithmSpec{
      [maxTFs, nIDCs, delay, cruStart = crus.front(), cruEnde = crus.back(), mIDCs, dropTFsRandom, rangeTFsDrop](ProcessingContext& ctx) {
        const auto tf = processing_helpers::getCurrentTF(ctx);

        if (!rangeTFsDrop.empty()) {
          if (tf >= rangeTFsDrop.front() && tf <= rangeTFsDrop.back()) {
            LOGP(info, "Dropping TF as specified from range: {}", tf);
            return;
          }
        }

        if (dropTFsRandom && !gRandom->Integer(dropTFsRandom)) {
          LOGP(info, "Dropping TF: {}", tf);
          return;
        }

        auto start = timer::now();
        const unsigned int additionalInterval = (tf % 3) ? 1 : 0; // in each integration inerval are either 10 or 11 values when having 128 orbits per TF and 12 orbits integration length
        const unsigned int intervals = (nIDCs + additionalInterval);
        const bool generateIDCs = mIDCs.front().empty();
        const float irVar = 1 + 0.1 * std::sin(tf * 0.0035); // IR variations

        std::vector<int> intervalsRand;
        std::vector<float> normFac;
        intervalsRand.reserve(intervals);
        normFac.reserve(intervals);
        const int nIntervalsMax = generateIDCs ? intervals : (mIDCs[0].size() / o2::tpc::Mapper::PADSPERREGION[0]);
        const float globalScaling = gRandom->Gaus(1, 0.2);
        for (int i = 0; i < intervals; ++i) {
          intervalsRand.emplace_back(gRandom->Integer(nIntervalsMax));
          normFac.emplace_back(globalScaling + gRandom->Gaus(1, 0.02));
        }

        for (uint32_t icru = cruStart; icru <= cruEnde; ++icru) {
          o2::tpc::CRU cruTmp(icru);
          const unsigned int nPads = o2::tpc::Mapper::PADSPERREGION[cruTmp.region()];
          const int cru = (icru + tf * Mapper::NREGIONS) % o2::tpc::CRU::MaxCRU; // shuffle CRUs
          o2::pmr::vector<float> idcs;
          idcs.reserve(generateIDCs ? o2::tpc::Mapper::PADSPERREGION[cruTmp.region()] : mIDCs[cru].size());
          const int nIntervals = intervalsRand.size();
          for (int interval = 0; interval < nIntervals; ++interval) {
            const int offset = intervalsRand[interval] * nPads;
            for (int iPad = 0; iPad < nPads; ++iPad) {
              if (generateIDCs) {
                idcs.emplace_back(irVar * gRandom->Gaus(10, 20) / normFac[interval]);
              } else {
                idcs.emplace_back(irVar * mIDCs[cru][offset + iPad] / normFac[interval]);
              }
            }
          }
          ctx.outputs().adoptContainer(Output{gDataOriginTPC, TPCIntegrateIDCDevice::getDataDescription(TPCIntegrateIDCDevice::IDCFormat::Sim), o2::header::DataHeader::SubSpecificationType{icru << 7}, Lifetime::Timeframe}, std::move(idcs));
        }

        if (delay) {
          auto stop = timer::now();
          auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
          float totalTime = milliseconds.count();
          const int sleepTime = (totalTime < intervals) ? (intervals - totalTime) : 0;
          if (!(tf % 100)) {
            LOGP(info, "time: {}  for {} intervals (ms): sleep for {} for TF: {}", totalTime, intervals, sleepTime, tf);
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        }

        if (tf >= (maxTFs - 1)) {
          ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
        }
      }}};
}

class TPCReceiveEPNSpec : public o2::framework::Task
{
  /// device for receiving and aggregating fourier coefficients from EPN device
 public:
  TPCReceiveEPNSpec(const unsigned int nTFs, const unsigned int nFourierCoefficients) : mFourierCoeffEPN(nTFs, o2::tpc::FourierCoeff(1, nFourierCoefficients)), mMaxTF{nTFs - 1} {};

  void run(o2::framework::ProcessingContext& ctx) final
  {
    const auto tf = processing_helpers::getCurrentTF(ctx);

    for (auto const& ref : InputRecordWalker(ctx.inputs(), mFilter)) {
      auto const* tpcFourierCoeffHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const int side = tpcFourierCoeffHeader->subSpecification;
      mFourierCoeffEPN[tf].mFourierCoefficients = ctx.inputs().get<std::vector<float>>(ref);
    }

    if (tf == mMaxTF) {
      ctx.outputs().snapshot(Output{gDataOriginTPC, getDataDescriptionCoeffEPN()}, mFourierCoeffEPN);
    }
  }

 private:
  std::vector<o2::tpc::FourierCoeff> mFourierCoeffEPN; ///< one set of fourier coefficients per TF
  const unsigned int mMaxTF{};                         ///< max TF when aggregator will send data
  const std::vector<InputSpec> mFilter = {{"coeffEPN", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFourierTransformEPNSpec::getDataDescription()}, Lifetime::Sporadic}};
};

class TPCCompareFourierCoeffSpec : public o2::framework::Task
{
  /// device for receiving the fourier coefficients from the EPN and the aggregator and comparing them
 public:
  void run(o2::framework::ProcessingContext& ctx) final
  {
    LOGP(info, "==== Comparing Fourier coefficients for EPN and Aggregator... ====");
    const auto fourierCoeffAgg = ctx.inputs().get<FourierCoeff*>(ctx.inputs().get("coeffAgg"));
    const std::vector<o2::tpc::FourierCoeff> fourierCoeffEPN = ctx.inputs().get<std::vector<o2::tpc::FourierCoeff>>(ctx.inputs().get("coeffEPN"));
    for (int tf = 0; tf < fourierCoeffEPN.size(); ++tf) {
      for (int i = 0; i < fourierCoeffEPN[tf].getNCoefficientsPerTF(); ++i) {
        const float epnval = fourierCoeffEPN[tf](i);
        const float aggval = (*fourierCoeffAgg)(fourierCoeffAgg->getIndex(tf, i));
        ASSERT_ERROR((std::abs(std::min(epnval, aggval)) < 1.f) ? isSameZero(epnval, aggval) : isSame(epnval, aggval));
      }
    }
    LOGP(info, "==== Test finished successfull! Fourier coefficients for EPN and Aggregator are equal. ====");
    ctx.services().get<ControlService>().endOfStream();
    ctx.services().get<ControlService>().readyToQuit(QuitRequest::All);
  }

 private:
  const float mTolerance = 1e-4; ///< max accepted tolerance between EPN and Aggregator

  bool isSameZero(const float epnval, const float aggval) const { return (epnval - aggval) * (epnval - aggval) < mTolerance; }
  bool isSame(const float epnval, const float aggval) const { return std::abs((epnval - aggval) / std::min(epnval, aggval)) < mTolerance; }
};

DataProcessorSpec receiveFourierCoeffEPN(const unsigned int maxTFs, const unsigned int nFourierCoefficients)
{
  std::vector<InputSpec> inputSpecs{{"coeffEPN", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFourierTransformEPNSpec::getDataDescription()}, Lifetime::Sporadic}};
  std::vector<OutputSpec> outputSpecs{ConcreteDataTypeMatcher{gDataOriginTPC, getDataDescriptionCoeffEPN()}};
  return DataProcessorSpec{
    "idc-fouriercoeff-epn",
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCReceiveEPNSpec>(maxTFs, nFourierCoefficients)},
  };
}

DataProcessorSpec compare_EPN_AGG()
{
  std::vector<InputSpec> inputSpecs;
  inputSpecs.emplace_back(InputSpec("coeffEPN", ConcreteDataTypeMatcher{gDataOriginTPC, getDataDescriptionCoeffEPN()}));
  inputSpecs.emplace_back(InputSpec("coeffAgg", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFourierTransformAggregatorSpec::getDataDescriptionFourier()}));
  return DataProcessorSpec{
    "idc-fouriercoeff-compare",
    inputSpecs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<TPCCompareFourierCoeffSpec>()},
  };
}
