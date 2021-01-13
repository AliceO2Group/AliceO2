// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DCStestWorkflow/DCSRandomDataGeneratorSpec.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DataPointGenerator.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include <TDatime.h>
#include <random>
#include <variant>
#include <string>
#include <algorithm>

using namespace o2::framework;

namespace
{
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
  std::mt19937 generator(std::random_device{}());
  std::vector<int> data;
  while (data.size() != size) {
    data.emplace_back(distribution(generator));
    std::sort(begin(data), end(data));
    auto last = std::unique(begin(data), end(data)); // make sure we do not duplicate
    data.erase(last, end(data));
  }
  std::shuffle(begin(data), end(data), generator);
  for (auto i = 0; i < data.size(); ++i) {
    LOG(INFO) << "Generating randomly DP at index " << data[i];
  }
  return data;
}

/** generate DCS data points.
  *
  * @param hints vector of HintType describing what to generate
  * @param fraction fraction of the generated aliases that are returned (1.0 by default)
  *
  * @returns a vector of DataPointCompositeObjects
  */
std::vector<o2::dcs::DataPointCompositeObject> generate(const std::vector<o2::dcs::test::HintType> hints,
                                                        float fraction = 1.0,
                                                        uint64_t tfid = 0)
{
  std::vector<o2::dcs::DataPointCompositeObject> dataPoints;

  TDatime d;
  auto dsec = d.Convert();
  dsec += tfid;
  d.Set(dsec);

  std::string refDate = d.AsString();

  auto GenerateVisitor = [refDate](const auto& t) {
    return o2::dcs::generateRandomDataPoints({t.aliasPattern}, t.minValue, t.maxValue, refDate);
  };

  for (const auto& hint : hints) {
    auto dpcoms = std::visit(GenerateVisitor, hint);
    for (auto dp : dpcoms) {
      dataPoints.push_back(dp);
    }
  }
  if (fraction < 1.0) {
    auto indices = generateIntegers(fraction * dataPoints.size(), 0, dataPoints.size() - 1);
    std::vector<o2::dcs::DataPointCompositeObject> tmp;
    tmp.swap(dataPoints);
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
 public:
  DCSRandomDataGenerator(std::vector<o2::dcs::test::HintType> hints, o2::header::DataDescription description);

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

 private:
  uint64_t mMaxTF;
  uint64_t mTFs = 0;
  uint64_t mMaxCyclesNoFullMap;
  float mDeltaFraction;
  std::vector<o2::dcs::test::HintType> mDataPointHints;
  o2::header::DataDescription mDataDescription;
};

DCSRandomDataGenerator::DCSRandomDataGenerator(std::vector<o2::dcs::test::HintType> hints,
                                               o2::header::DataDescription description) : mDataPointHints(hints),
                                                                                          mDataDescription(description) {}

void DCSRandomDataGenerator::init(o2::framework::InitContext& ic)
{
  mMaxTF = ic.options().get<int64_t>("max-timeframes");
  mDeltaFraction = ic.options().get<float>("delta-fraction");
  mMaxCyclesNoFullMap = ic.options().get<int64_t>("max-cycles-no-full-map");
}

void DCSRandomDataGenerator::run(o2::framework::ProcessingContext& pc)
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

  TDatime d;
  auto dpcoms = generate(mDataPointHints, fraction, tfid);

  LOG(INFO) << "***************** TF " << tfid << " has generated " << dpcoms.size() << " DPs for TOF";
  pc.outputs().snapshot(Output{"DCS", mDataDescription, 0, Lifetime::Timeframe}, dpcoms);
  mTFs++;
}
} // namespace

namespace o2::dcs::test
{
o2::framework::DataProcessorSpec getDCSRandomDataGeneratorSpec(std::vector<o2::dcs::test::HintType> hints,
                                                               const char* detName)
{
  std::string desc{detName};
  desc += "DATAPOINTS";

  o2::header::DataDescription dd;

  dd.runtimeInit(desc.c_str(), desc.size());

  return DataProcessorSpec{
    "dcs-random-data-generator",
    Inputs{},
    Outputs{{{"outputDCS"}, "DCS", dd}},
    AlgorithmSpec{adaptFromTask<DCSRandomDataGenerator>(hints, dd)},
    Options{
      {"max-timeframes", VariantType::Int64, 99999999999ll, {"max TimeFrames to generate"}},
      {"delta-fraction", VariantType::Float, 0.05f, {"fraction of data points to put in the delta"}},
      {"max-cycles-no-full-map", VariantType::Int64, 6000ll, {"max num of cycles between the sending of 2 full maps"}}}};
}
} // namespace o2::dcs::test
