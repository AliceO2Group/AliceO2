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

#include <boost/throw_exception.hpp>
#include "Framework/DataRefUtils.h"
#include "Framework/ControlService.h"
#include "Algorithm/RangeTokenizer.h"
#include "TRandom.h"

#include <unistd.h>

using namespace o2::framework;

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(fatal) << R"(Test condition ")" #condition R"(" failed)"; \
  }

DataProcessorSpec generateIDCsCRU(int lane, const unsigned int maxTFs, const std::vector<uint32_t>& crus, const bool slowgen);
DataProcessorSpec ftAggregatorIDC(const unsigned int nFourierCoefficients, const unsigned int rangeIDC, const unsigned int maxTFs, const bool debug);
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
    {"fast-gen", VariantType::Bool, false, {"fast generation of IDCs by setting fixed IDC value"}},
    {"debug", VariantType::Bool, false, {"create debug for FT"}},
    {"idc-gen-lanes", VariantType::Int, 1, {"number of parallel lanes for generation of IDCs"}},
    {"idc-gen-time-lanes", VariantType::Int, 1, {"number of parallel lanes for generation of IDCs"}},
    {"nthreads", VariantType::Int, 1, {"Number of threads."}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  const auto tpcCRUs = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("crus"));
  const auto nCRUs = tpcCRUs.size();
  const auto first = tpcCRUs.begin();
  const auto last = std::min(tpcCRUs.end(), first + nCRUs);
  const std::vector<uint32_t> crus(first, last);
  const auto timeframes = static_cast<unsigned int>(config.options().get<int>("timeframes"));
  const auto iondrifttime = static_cast<unsigned int>(config.options().get<int>("ion-drift-time"));
  const auto nFourierCoefficients = std::clamp(static_cast<unsigned int>(config.options().get<int>("nFourierCoeff")), static_cast<unsigned int>(0), iondrifttime + 2);
  const auto nthreads = static_cast<unsigned int>(config.options().get<int>("nthreads"));
  const auto seed = static_cast<unsigned int>(config.options().get<int>("seed"));
  const auto idcgenlanes = static_cast<unsigned int>(config.options().get<int>("idc-gen-lanes"));
  const auto idcgentimelanes = static_cast<unsigned int>(config.options().get<int>("idc-gen-time-lanes"));

  const bool fft = config.options().get<bool>("use-naive-fft");
  const bool onlyIDCGen = config.options().get<bool>("only-idc-gen");
  const bool fastgen = config.options().get<bool>("fast-gen");
  const bool debugFT = config.options().get<bool>("debug");
  const unsigned int firstTF = 0;
  const unsigned int nLanes = 1;
  const bool loadFromFile = false;
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
    workflow.emplace_back(timePipeline(generateIDCsCRU(ilane, timeframes, rangeCRUs, fastgen), idcgentimelanes));
  }

  if (onlyIDCGen == true) {
    return workflow;
  }

  auto workflowTmp = WorkflowSpec{getTPCFLPIDCSpec<TPCFLPIDCDeviceNoGroup>(0, crus, iondrifttime, debugFT, false, false, "", true), getTPCFourierTransformEPNSpec(crus, iondrifttime, nFourierCoefficients, debugFT), getTPCDistributeIDCSpec(0, crus, timeframes, nLanes, firstTF, loadFromFile), ftAggregatorIDC(nFourierCoefficients, iondrifttime, timeframes, debugFT), receiveFourierCoeffEPN(timeframes, nFourierCoefficients), compare_EPN_AGG()};
  for (auto& spec : workflowTmp) {
    workflow.emplace_back(spec);
  }

  IDCAverageGroup<IDCAverageGroupTPC>::setNThreads(nthreads);
  TPCFourierTransformEPNSpec::IDCFType::setFFT(!fft);
  TPCFourierTransformAggregatorSpec::IDCFType::setNThreads(nthreads);
  TPCFourierTransformAggregatorSpec::IDCFType::setFFT(!fft);
  return workflow;
}

DataProcessorSpec generateIDCsCRU(int lane, const unsigned int maxTFs, const std::vector<uint32_t>& crus, const bool fastgen)
{
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

  return DataProcessorSpec{
    fmt::format("idc-generator-{:02}", lane).data(),
    Inputs{},
    outputSpecs,
    AlgorithmSpec{
      [maxTFs, fastgen, nIDCs, cruStart = crus.front(), cruEnde = crus.back()](ProcessingContext& ctx) {
        const auto tf = ctx.services().get<o2::framework::TimingInfo>().tfCounter;

        for (uint32_t icru = cruStart; icru <= cruEnde; ++icru) {
          const o2::tpc::CRU cruTmp(icru);
          const unsigned int nPads = o2::tpc::Mapper::PADSPERREGION[cruTmp.region()];
          const unsigned int additionalInterval = (tf % 3) ? 1 : 0; // in each integration inerval are either 10 or 11 values when having 128 orbits per TF and 12 orbits integration length
          const unsigned int nTotIDCs = (nIDCs + additionalInterval) * nPads;

          // generate random IDCs per CRU
          if (!fastgen) {
            o2::pmr::vector<float> idcs;
            idcs.reserve(nTotIDCs);
            for (int i = 0; i < nTotIDCs; ++i) {
              idcs.emplace_back(gRandom->Gaus(1000, 20));
            }
            ctx.outputs().adoptContainer(Output{gDataOriginTPC, TPCIntegrateIDCDevice::getDataDescription(TPCIntegrateIDCDevice::IDCFormat::Sim), o2::header::DataHeader::SubSpecificationType{icru << 7}, Lifetime::Timeframe}, std::move(idcs));
          } else {
            o2::pmr::vector<float> idcs(nTotIDCs, 1000.);
            ctx.outputs().adoptContainer(Output{gDataOriginTPC, TPCIntegrateIDCDevice::getDataDescription(TPCIntegrateIDCDevice::IDCFormat::Sim), o2::header::DataHeader::SubSpecificationType{icru << 7}, Lifetime::Timeframe}, std::move(idcs));
          }
        }

        if (tf >= (maxTFs - 1)) {
          ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
        }
      }}};
}

DataProcessorSpec ftAggregatorIDC(const unsigned int nFourierCoefficients, const unsigned int rangeIDC, const unsigned int maxTFs, const bool debug)
{
  std::vector<ConfigParamSpec> options{
    {"ccdb-uri", VariantType::String, "", {"URI for the CCDB access."}},
    {"update-not-grouping-parameter", VariantType::Bool, true, {"Do NOT Update/Writing grouping parameters to CCDB."}}};

  auto spec = getTPCFourierTransformAggregatorSpec(maxTFs, rangeIDC, nFourierCoefficients, debug, true);
  std::swap(spec.options, options);

  return spec;
}

class TPCReceiveEPNSpec : public o2::framework::Task
{
  /// device for receiving and aggregating fourier coefficients from EPN device
 public:
  TPCReceiveEPNSpec(const unsigned int nTFs, const unsigned int nFourierCoefficients) : mFourierCoeffEPN(nTFs, o2::tpc::FourierCoeff(1, nFourierCoefficients)), mMaxTF{nTFs - 1} {};

  void run(o2::framework::ProcessingContext& ctx) final
  {
    const auto& tinfo = ctx.services().get<o2::framework::TimingInfo>();
    for (int i = 0; i < o2::tpc::SIDES; ++i) {
      const DataRef ref = ctx.inputs().getByPos(i);
      auto const* tpcFourierCoeffHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const int side = tpcFourierCoeffHeader->subSpecification;
      const auto tf = tinfo.tfCounter;
      mFourierCoeffEPN[tf].mFourierCoefficients = ctx.inputs().get<std::vector<float>>(ref);
    }

    const auto tf = tinfo.tfCounter;
    if (tf == mMaxTF) {
      ctx.outputs().snapshot(Output{gDataOriginTPC, getDataDescriptionCoeffEPN()}, mFourierCoeffEPN);
    }
  }

 private:
  std::vector<o2::tpc::FourierCoeff> mFourierCoeffEPN; ///< one set of fourier coefficients per TF
  const unsigned int mMaxTF{};                         ///< max TF when aggregator will send data
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
    for (int iside = 0; iside < o2::tpc::SIDES; ++iside) {
      const o2::tpc::Side side = iside == 0 ? o2::tpc::Side::A : o2::tpc::Side::C;
      for (int tf = 0; tf < fourierCoeffEPN.size(); ++tf) {
        for (int i = 0; i < fourierCoeffEPN[tf].getNCoefficientsPerTF(); ++i) {
          const float epnval = fourierCoeffEPN[tf](i);
          const float aggval = (*fourierCoeffAgg)(fourierCoeffAgg->getIndex(tf, i));
          ASSERT_ERROR((std::abs(std::min(epnval, aggval)) < 1.f) ? isSameZero(epnval, aggval) : isSame(epnval, aggval));
        }
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
  // generate IDCs per CRU
  std::vector<InputSpec> inputSpecs;
  inputSpecs.emplace_back(InputSpec{"coeffAEPN", gDataOriginTPC, TPCFourierTransformEPNSpec::getDataDescription(), o2::header::DataHeader::SubSpecificationType{o2::tpc::Side::A}, Lifetime::Timeframe});
  inputSpecs.emplace_back(InputSpec{"coeffCEPN", gDataOriginTPC, TPCFourierTransformEPNSpec::getDataDescription(), o2::header::DataHeader::SubSpecificationType{o2::tpc::Side::C}, Lifetime::Timeframe});
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
  // generate IDCs per CRU
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
