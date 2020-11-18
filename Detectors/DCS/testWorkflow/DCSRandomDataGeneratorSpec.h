// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_DCS_RANDOM_DATA_GENERATOR_SPEC_H
#define O2_DCS_RANDOM_DATA_GENERATOR_SPEC_H

#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointGenerator.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include <random>
#include <variant>
#include <string>

using namespace o2::framework;

namespace
{
/*
 * A compact representation a group of alias to be generated
 */
template <typename T>
struct DataPointHint {
  std::string aliasPattern; // alias pattern e.g. DET/HV/Crate[0..2]/Channel[000..012]/vMon
  T minValue;               // minimum value to generate
  T maxValue;               // maximum value to generate
};

using HintType = std::variant<DataPointHint<double>,
                              DataPointHint<uint32_t>,
                              DataPointHint<int32_t>,
                              DataPointHint<char>,
                              DataPointHint<bool>,
                              DataPointHint<std::string>>;

/** generate random integers uniformly distributed within a range.
  *
  * @param size the number of integers to generate
  * @param min the minimum value to be generated
  * @param max the maximum value to be generated
  *
  * @returns a vector of integers
  */
std::vector<int> generateIntegers(size_t size, int min, int max)
{
  std::uniform_int_distribution<int> distribution(min, max);
  std::mt19937 generator;
  std::vector<int> data(size);
  std::generate(data.begin(), data.end(), [&]() { return distribution(generator); });
  return data;
}

/** generate DCS data points.
  *
  * @param hints vector of HintType describing what to generate
  * @param fraction fraction of the generated aliases that are returned (1.0 by default)
  *
  * @returns a vector of DataPointCompositeObjects
  */
std::vector<o2::dcs::DataPointCompositeObject> generate(const std::vector<HintType> hints,
                                                        float fraction = 1.0)
{
  std::vector<o2::dcs::DataPointCompositeObject> dataPoints;

  auto GenerateVisitor = [](const auto& t) {
    return o2::dcs::generateRandomDataPoints({t.aliasPattern}, t.minValue, t.maxValue);
  };

  for (const auto& hint : hints) {
    auto dpcoms = std::visit(GenerateVisitor, hint);
    for (auto dp : dpcoms) {
      dataPoints.push_back(dp);
    }
  }
  if (fraction < 1.0) {
    auto indices = generateIntegers(fraction * dataPoints.size(), 0, dataPoints.size() - 1);
    auto tmp = dataPoints;
    dataPoints.clear();
    for (auto i : indices) {
      dataPoints.push_back(tmp[i]);
    }
  }
  return dataPoints;
}

/** 
  * DCSRandomDataGenerator is an example device that generates random 
  * DCS Data Points.
  *
  * The actual description of what is generated is hard-coded in 
  * the init() method.
  */
class DCSRandomDataGenerator : public o2::framework::Task
{
  using DPID = o2::dcs::DataPointIdentifier;
  using DPVAL = o2::dcs::DataPointValue;
  using DPCOM = o2::dcs::DataPointCompositeObject;

 public:
  void init(o2::framework::InitContext& ic) final
  {
    mMaxTF = ic.options().get<int64_t>("max-timeframes");
    mDeltaFraction = ic.options().get<float>("delta-fraction");
    mMaxCyclesNoFullMap = ic.options().get<int64_t>("max-cycles-no-full-map");

    // create the list of DataPointHints to be used by the generator
    mDataPointHints.emplace_back(DataPointHint<char>{"TestChar_0", 'A', 'z'});
    mDataPointHints.emplace_back(DataPointHint<double>{"TestDouble_[0..3]", 0, 1700});
    mDataPointHints.emplace_back(DataPointHint<int32_t>{"TestInt_[0..50000{:d}]", 0, 1234});
    mDataPointHints.emplace_back(DataPointHint<bool>{"TestBool_[00..03]", 0, 1});
    mDataPointHints.emplace_back(DataPointHint<std::string>{"TestString_0", "ABC", "ABCDEF"});
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto input = pc.inputs().begin();
    uint64_t tfid = o2::header::get<o2::framework::DataProcessingHeader*>((*input).header)->startTime;
    if (tfid >= mMaxTF) {
      LOG(INFO) << "Data generator reached TF " << tfid << ", stopping";
      pc.services().get<o2::framework::ControlService>().endOfStream();
      pc.services().get<o2::framework::ControlService>().readyToQuit(o2::framework::QuitRequest::Me);
    }

    bool generateFBI = (mTFs % mMaxCyclesNoFullMap == 0);
    // fraction is one if we generate FBI (Full Buffer Image)
    float fraction = (generateFBI ? 1.0 : mDeltaFraction);

    auto dpcoms = generate(mDataPointHints, fraction);

    // the output must always get both FBI and Delta, but one of them is empty.
    std::vector<o2::dcs::DataPointCompositeObject> empty;
    pc.outputs().snapshot(Output{"DCS", "DATAPOINTS", 0, Lifetime::Timeframe}, generateFBI ? dpcoms : empty);
    pc.outputs().snapshot(Output{"DCS", "DATAPOINTSdelta", 0, Lifetime::Timeframe}, generateFBI ? empty : dpcoms);
    mTFs++;
  }

 private:
  uint64_t mMaxTF;
  uint64_t mTFs = 0;
  uint64_t mMaxCyclesNoFullMap;
  float mDeltaFraction;
  std::vector<HintType> mDataPointHints;
};

} // namespace

DataProcessorSpec getDCSRandomDataGeneratorSpec()
{
  return DataProcessorSpec{
    "dcs-random-data-generator",
    Inputs{},
    Outputs{{{"outputDCS"}, "DCS", "DATAPOINTS"}, {{"outputDCSdelta"}, "DCS", "DATAPOINTSdelta"}},
    AlgorithmSpec{adaptFromTask<DCSRandomDataGenerator>()},
    Options{
      {"max-timeframes", VariantType::Int64, 99999999999ll, {"max TimeFrames to generate"}},
      {"delta-fraction", VariantType::Float, 0.05f, {"fraction of data points to put in the delta"}},
      {"max-cycles-no-full-map", VariantType::Int64, 6000ll, {"max num of cycles between the sending of 2 full maps"}}}};
}

#endif
