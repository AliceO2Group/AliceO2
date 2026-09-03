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

/// \file  test_cmv_generator.cxx
/// \brief DPL source workflow that generates dummy CMV data for testing the CMV FLP pipeline.
///
/// Replaces o2-tpc-cmv-to-vector in tests; directly emits CMVVECTOR and CMVORBITS
/// messages per CRU per TF so the workflow can be piped straight into o2-tpc-cmv-flp:
///
///   o2-tpc-cmv-test-generator --crus 0-359 --timeframes 100 \
///   | o2-tpc-cmv-flp --crus 0-359 --n-TFs-buffer 10        \
///   | o2-dpl-output-proxy --dataspec "downstream:TPC/CMVGROUP;downstream:TPC/CMVORBITINFO" ...
///
/// \author Ernst Hellbar <ernst.hellbar@cern.ch>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/Logger.h"
#include "Headers/DataHeader.h"
#include "Algorithm/RangeTokenizer.h"
#include "TPCBase/CRU.h"
#include "DataFormatsTPC/CMV.h"
#include "TPCCalibration/CMVHelper.h"
#include "TPCCalibration/CMVContainer.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DetectorsRaw/HBFUtils.h"
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <memory>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>

using namespace o2::framework;
using o2::header::gDataOriginTPC;

// ─────────────────────────────────────────────────────────────────────────────
// workflow options
// ─────────────────────────────────────────────────────────────────────────────
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  const std::string cruDefault = "0-" + std::to_string(o2::tpc::CRU::MaxCRU - 1);
  std::vector<ConfigParamSpec> options{
    {"crus", VariantType::String, cruDefault.c_str(), {"List of CRUs, comma-separated ranges, e.g. 0-3,7,9-15"}},
    {"timeframes", VariantType::Int, 100, {"Number of TFs to generate; use -1 to run indefinitely"}},
    {"delay", VariantType::Bool, false, {"Add delay after sending all CRUs"}},
    {"delayTime", VariantType::Int, 1, {"Duration of the global per-TF delay in ms (requires --delay true)"}},
    {"delayEveryN", VariantType::Int, 1, {"Apply the global delay only on average once every N TFs, randomly chosen (1 = every TF, requires --delay true)"}},
    {"delayCRUs", VariantType::String, "", {"CRUs for which to add an extra per-CRU delay before sending, comma-separated ranges"}},
    {"delayTimeCRUs", VariantType::Int, 1, {"Duration of the per-CRU delay in ms (requires --delayCRUs)"}},
    {"dropTFsRandom", VariantType::Int, 0, {"Drop a whole TF randomly: on average one every N TFs (0 = disabled)"}},
    {"dropTFsRange", VariantType::String, "", {"Drop all TFs in this range, e.g. 10-12"}},
    {"tfLength", VariantType::Float, 0.f, {"Minimum wall-clock time between consecutively sent TFs in ms (rate limiter); the generator sleeps if a TF is produced faster than this (0 = disabled)"}},
    {"seed", VariantType::Int, 42, {"RNG seed for CMV value generation"}},
    {"amplitude", VariantType::Float, 5.0f, {"Amplitude of the sinusoidal CMV signal (ADC units); ignored when --input-file is set"}},
    {"noise", VariantType::Float, 1.0f, {"Gaussian noise std-dev added per time bin (ADC units); used as the smearing width in --input-file mode"}},
    {"input-file", VariantType::String, "", {"ROOT file with a CMV 'ccdb_object' tree; the template TF (see --input-entry) is decoded once and re-emitted, smeared per generated TF. Empty = synthetic sinusoidal signal"}},
    {"input-entry", VariantType::Int, 0, {"Tree entry (TF index) used as the template when --input-file is set"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options, "hbfutils");
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

// ─────────────────────────────────────────────────────────────────────────────
// generator device
// ─────────────────────────────────────────────────────────────────────────────
class CMVGeneratorDevice : public o2::framework::Task
{
 public:
  static constexpr uint32_t sOrbitsPerPacket = 8; ///< each CMV packet covers 8 heartbeat orbits

  CMVGeneratorDevice(const std::vector<uint32_t>& crus,
                     const std::unordered_set<uint32_t>& delayCRUs,
                     unsigned int maxTFs,
                     bool delay,
                     int delayTime,
                     int delayEveryN,
                     int delayTimeCRUs,
                     int dropTFsRandom,
                     const std::vector<int>& rangeTFsDrop,
                     float tfLength,
                     float amplitude,
                     float noise,
                     int seed,
                     const std::string& inputFile,
                     long long inputEntry)
    : mCRUs(crus), mDelayCRUs(delayCRUs), mMaxTFs(maxTFs), mDelay(delay), mDelayTime(delayTime), mDelayEveryN(delayEveryN), mDelayTimeCRUs(delayTimeCRUs), mDropTFsRandom(dropTFsRandom), mRangeTFsDrop(rangeTFsDrop), mTFLength(tfLength), mAmplitude(amplitude), mNoise(noise), mRng(static_cast<std::mt19937::result_type>(seed)), mInputFileName(inputFile), mInputEntry(inputEntry) {}

  void init(o2::framework::InitContext& ic) final
  {
    mTimer100TFs = std::chrono::high_resolution_clock::now();
    mLastTFTime = std::chrono::high_resolution_clock::now();

    if (!mCRUs.empty()) {
      LOGP(info, "crus: {}", fmt::join(mCRUs, ", "));
    }
    if (!mDelayCRUs.empty()) {
      const std::vector<uint32_t> delayCRUsSorted(mDelayCRUs.begin(), mDelayCRUs.end());
      LOGP(info, "delayCRUs: {}", fmt::join(delayCRUsSorted, ", "));
    }

    mWriteDebug = ic.options().get<bool>("write-debug");
    if (mWriteDebug) {
      mDebugStreamFileName = ic.options().get<std::string>("debug-file-name");
      LOGP(info, "Creating debug stream {}", mDebugStreamFileName);
      mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugStreamFileName.data(), "recreate");
    }

    if (!mInputFileName.empty()) {
      o2::tpc::CMVFileHandle handle;
      if (!handle.open(mInputFileName)) {
        throw std::runtime_error("CMV generator: failed to open input file " + mInputFileName);
      }
      const auto nEntries = handle.tree->GetEntries();
      if (mInputEntry < 0 || mInputEntry >= nEntries) {
        const auto msg = fmt::format("CMV generator: --input-entry {} out of range [0, {}) in {}", mInputEntry, nEntries, mInputFileName);
        handle.close();
        throw std::runtime_error(msg);
      }
      const o2::tpc::CMVPerTF* tmpl = handle.getEntry(mInputEntry);
      if (!tmpl) {
        handle.close();
        throw std::runtime_error("CMV generator: failed to read/decode entry from " + mInputFileName);
      }
      // When noise is enabled we keep the per-CRU float template and re-encode it
      // (template + noise) every TF. When noise is disabled the output is identical
      // for every TF, so we encode it once here and just re-snapshot it in run().
      const bool addNoise = (mNoise > 0.f);
      if (addNoise) {
        mBaseCMVFloat.resize(mCRUs.size());
        for (size_t iCRU = 0; iCRU < mCRUs.size(); ++iCRU) {
          const auto cru = mCRUs[iCRU];
          auto& base = mBaseCMVFloat[iCRU];
          base.resize(o2::tpc::cmv::NTimeBinsPerTF);
          for (uint32_t tb = 0; tb < o2::tpc::cmv::NTimeBinsPerTF; ++tb) {
            base[tb] = tmpl->getCMVFloat(static_cast<int>(cru), static_cast<int>(tb));
          }
        }
      } else {
        mBaseCMVEncoded.resize(mCRUs.size());
        for (size_t iCRU = 0; iCRU < mCRUs.size(); ++iCRU) {
          const auto cru = mCRUs[iCRU];
          auto& enc = mBaseCMVEncoded[iCRU];
          enc.resize(o2::tpc::cmv::NTimeBinsPerTF);
          for (uint32_t tb = 0; tb < o2::tpc::cmv::NTimeBinsPerTF; ++tb) {
            o2::tpc::cmv::Data d;
            d.setCMVFloat(tmpl->getCMVFloat(static_cast<int>(cru), static_cast<int>(tb)));
            enc[tb] = d.getCMV();
          }
        }
      }
      handle.close();
      mUseInputFile = true;
      LOGP(info, "Loaded CMV template from {} (entry {}): {} CRUs x {} bins, noise sigma {} ADC ({})",
           mInputFileName, mInputEntry, mCRUs.size(), o2::tpc::cmv::NTimeBinsPerTF, mNoise,
           addNoise ? "re-smeared per TF" : "encoded once, replayed verbatim");
    }
  }

  void run(o2::framework::ProcessingContext& ctx) final
  {
    using timer = std::chrono::high_resolution_clock;
    const auto tf = o2::tpc::processing_helpers::getCurrentTF(ctx);

    // ── TF dropping ──────────────────────────────────────────────────────────
    // Note: RangeTokenizer guarantees sorted output, so front()/back() are min/max.
    if (!mRangeTFsDrop.empty() && tf >= static_cast<uint32_t>(mRangeTFsDrop.front()) && tf <= static_cast<uint32_t>(mRangeTFsDrop.back())) {
      LOGP(info, "Dropping TF {} (range drop)", tf);
      return;
    }
    if (mDropTFsRandom > 0 && std::uniform_int_distribution<int>{0, mDropTFsRandom - 1}(mRng) == 0) {
      LOGP(info, "Dropping TF {} (random drop)", tf);
      return;
    }

    auto start = timer::now();

    // ── CMV values ───────────────────────────────────────────────────────────
    // NTimeBinsPerTF = NPacketsPerTFPerCRU (4) * NTimeBinsPerPacket (3564) = 14256
    //   - synthetic mode: shared cmvVec = sinusoidal signal + noise (same for all CRUs)
    //   - input-file mode: per-CRU template + the shared noise vector
    const bool addNoise = (mNoise > 0.f); // skip all RNG when --noise 0
    std::normal_distribution<float> noiseDist{0.f, mNoise};
    std::vector<uint16_t> cmvVec(o2::tpc::cmv::NTimeBinsPerTF);
    std::vector<float> noiseVec; // only populated in input-file mode when noise is enabled
    if (mUseInputFile) {
      if (addNoise) {
        noiseVec.resize(o2::tpc::cmv::NTimeBinsPerTF);
        for (auto& n : noiseVec) {
          n = noiseDist(mRng);
        }
      }
    } else {
      const float signal = -std::abs(mAmplitude * std::sin(tf * 0.05f));
      for (auto& v : cmvVec) {
        o2::tpc::cmv::Data d;
        d.setCMVFloat(addNoise ? (signal + noiseDist(mRng)) : signal);
        v = d.getCMV();
      }
    }

    // ── Orbit / BC info (same for all CRUs) ──────────────────────────────────
    // One packed (orbit<<32|bc) entry per CMV packet (4 per TF).
    // Each packet covers 8 heartbeat orbits (NTimeBinsPerPacket = 3564 = 8 LHC orbits),
    // so the orbit advances by 8 per packet and by NPacketsPerTFPerCRU*8 = 32 per TF.
    std::vector<uint64_t> orbitBCVec(o2::tpc::cmv::NPacketsPerTFPerCRU);
    for (uint32_t pkt = 0; pkt < o2::tpc::cmv::NPacketsPerTFPerCRU; ++pkt) {
      const uint32_t orbit = static_cast<uint32_t>(tf * o2::tpc::cmv::NPacketsPerTFPerCRU * sOrbitsPerPacket + pkt * sOrbitsPerPacket);
      orbitBCVec[pkt] = uint64_t(orbit) << 32; // bc = 0
    }

    for (size_t iCRU = 0; iCRU < mCRUs.size(); ++iCRU) {
      const auto cru = mCRUs[iCRU];
      const o2::header::DataHeader::SubSpecificationType subSpec{cru << 7};

      // ── per-CRU delay ────────────────────────────────────────────────────
      if (mDelayCRUs.count(cru)) {
        LOGP(info, "Delaying CRU {} by {} ms (TF {})", cru, mDelayTimeCRUs, tf);
        std::this_thread::sleep_for(std::chrono::milliseconds(mDelayTimeCRUs));
      }

      // Select the vector to emit: the precomputed template (no noise) is sent
      // verbatim; otherwise this CRU's template is smeared with the shared noise.
      std::vector<uint16_t>* out = &cmvVec;
      if (mUseInputFile && !addNoise) {
        out = &mBaseCMVEncoded[iCRU]; // encoded once in init(), reused every TF
      } else if (mUseInputFile) {
        const auto& base = mBaseCMVFloat[iCRU];
        for (uint32_t tb = 0; tb < o2::tpc::cmv::NTimeBinsPerTF; ++tb) {
          o2::tpc::cmv::Data d;
          d.setCMVFloat(base[tb] + noiseVec[tb]);
          cmvVec[tb] = d.getCMV();
        }
      }

      ctx.outputs().snapshot(Output{gDataOriginTPC, "CMVVECTOR", subSpec}, *out);
      ctx.outputs().snapshot(Output{gDataOriginTPC, "CMVORBITS", subSpec}, orbitBCVec);

      if (mWriteDebug) {
        auto& stream = (*mDebugStream) << "cmvs";
        stream << "cru=" << cru
               << "tfCounter=" << tf
               << "nCMVs=" << out->size()
               << "cmvs=" << *out
               << "\n";
      }
    }

    if (!(tf % 100)) {
      const auto elapsed100 = std::chrono::duration_cast<std::chrono::milliseconds>(timer::now() - mTimer100TFs).count();
      LOGP(info, "Generated CMV data for TF {} ({} ms for last 100 TFs)", tf, elapsed100);
      mTimer100TFs = timer::now();
    }

    // ── global delay ─────────────────────────────────────────────────────────
    if (mDelay && (mDelayEveryN <= 1 || std::uniform_int_distribution<int>{0, mDelayEveryN - 1}(mRng) == 0)) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(timer::now() - start).count();
      if (elapsed < mDelayTime) {
        LOGP(info, "Delaying TF {} by {} ms", tf, mDelayTime - elapsed);
        std::this_thread::sleep_for(std::chrono::milliseconds(mDelayTime - elapsed));
      }
    }

    // ── rate limiting ────────────────────────────────────────────────────────
    // Enforce a minimum wall-clock spacing between consecutively sent TFs.
    if (mTFLength > 0.f) {
      const auto elapsedSinceLast = std::chrono::duration_cast<std::chrono::microseconds>(timer::now() - mLastTFTime).count();
      const auto tfLengthUs = static_cast<int64_t>(mTFLength * 1000.f);
      if (elapsedSinceLast < tfLengthUs) {
        const auto waitUs = tfLengthUs - elapsedSinceLast;
        LOGP(info, "Rate limiting TF {}: waiting {} us (tfLength={} ms)", tf, waitUs, mTFLength);
        std::this_thread::sleep_for(std::chrono::microseconds(waitUs));
      }
      mLastTFTime = timer::now();
    }

    // endOfStream() propagates the EoS signal to downstream devices (required for source devices).
    if (mMaxTFs != std::numeric_limits<unsigned int>::max() && tf >= mMaxTFs - 1) {
      ctx.services().get<ControlService>().endOfStream();
      ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext&) final { closeFiles(); }
  void stop() final { closeFiles(); }

 private:
  void closeFiles()
  {
    if (mDebugStream) {
      auto& stream = (*mDebugStream) << "cmvs";
      auto& tree = stream.getTree();
      tree.SetAlias("sector", "int(cru/10)");
      mDebugStream->Close();
      mDebugStream.reset(nullptr);
    }
  }

  const std::vector<uint32_t> mCRUs{};
  const std::unordered_set<uint32_t> mDelayCRUs{};
  const unsigned int mMaxTFs{};
  const bool mDelay{false};
  const int mDelayTime{1};
  const int mDelayEveryN{1};
  const int mDelayTimeCRUs{1};
  const int mDropTFsRandom{0};
  const std::vector<int> mRangeTFsDrop{};
  const float mTFLength{0.f};
  const float mAmplitude{5.f};
  const float mNoise{1.f};
  std::mt19937 mRng{};
  const std::string mInputFileName{};                 ///< CMV ROOT file to use as template ("" = synthetic mode)
  const long long mInputEntry{0};                     ///< tree entry (TF) used as template
  bool mUseInputFile{false};                          ///< true once a template has been loaded
  std::vector<std::vector<float>> mBaseCMVFloat;      ///< decoded template CMV values [iCRU][timeBin] (noise>0 path), aligned to mCRUs
  std::vector<std::vector<uint16_t>> mBaseCMVEncoded; ///< pre-encoded template output [iCRU][timeBin] (noise==0 path), aligned to mCRUs
  std::chrono::high_resolution_clock::time_point mTimer100TFs{};
  std::chrono::high_resolution_clock::time_point mLastTFTime{};
  bool mWriteDebug{false};
  std::string mDebugStreamFileName{};
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugStream{};
};

// ─────────────────────────────────────────────────────────────────────────────
DataProcessorSpec generateCMVsCRU(const std::vector<uint32_t>& crus,
                                  const std::unordered_set<uint32_t>& delayCRUs,
                                  unsigned int maxTFs,
                                  bool delay,
                                  int delayTime,
                                  int delayEveryN,
                                  int delayTimeCRUs,
                                  int dropTFsRandom,
                                  const std::vector<int>& rangeTFsDrop,
                                  float tfLength,
                                  float amplitude,
                                  float noise,
                                  int seed,
                                  const std::string& inputFile,
                                  long long inputEntry)
{
  std::vector<OutputSpec> outputSpecs;
  outputSpecs.reserve(crus.size() * 2);
  for (const auto cru : crus) {
    const o2::header::DataHeader::SubSpecificationType subSpec{cru << 7};
    outputSpecs.emplace_back(gDataOriginTPC, "CMVVECTOR", subSpec, Lifetime::Timeframe);
    outputSpecs.emplace_back(gDataOriginTPC, "CMVORBITS", subSpec, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "tpc-cmv-generator",
    Inputs{},
    outputSpecs,
    AlgorithmSpec{adaptFromTask<CMVGeneratorDevice>(crus, delayCRUs, maxTFs, delay, delayTime, delayEveryN, delayTimeCRUs, dropTFsRandom, rangeTFsDrop, tfLength, amplitude, noise, seed, inputFile, inputEntry)},
    Options{
      {"write-debug", VariantType::Bool, false, {"Write a debug output tree"}},
      {"debug-file-name", VariantType::String, "./cmv_generator_debug.root", {"Name of the debug output file"}},
    }};
}

// ─────────────────────────────────────────────────────────────────────────────
WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  const auto tpcCRUs = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("crus"));
  const std::vector<uint32_t> crus(tpcCRUs.begin(), tpcCRUs.end());

  const auto delayCRUsStr = config.options().get<std::string>("delayCRUs");
  std::unordered_set<uint32_t> delayCRUs;
  if (!delayCRUsStr.empty()) {
    for (const auto cru : o2::RangeTokenizer::tokenize<int>(delayCRUsStr)) {
      delayCRUs.insert(static_cast<uint32_t>(cru));
    }
  }

  const auto dropTFsRangeStr = config.options().get<std::string>("dropTFsRange");
  const auto rangeTFsDrop = dropTFsRangeStr.empty() ? std::vector<int>{} : o2::RangeTokenizer::tokenize<int>(dropTFsRangeStr);
  const int timeframesInt = config.options().get<int>("timeframes");
  // -1 means run indefinitely; map to UINT_MAX so the termination check never fires.
  const auto timeframes = (timeframesInt < 0) ? std::numeric_limits<unsigned int>::max() : static_cast<unsigned int>(timeframesInt);
  const auto delay = config.options().get<bool>("delay");
  const auto delayTime = config.options().get<int>("delayTime");
  const auto delayEveryN = config.options().get<int>("delayEveryN");
  const auto delayTimeCRUs = config.options().get<int>("delayTimeCRUs");
  const auto dropTFsRandom = config.options().get<int>("dropTFsRandom");
  const auto tfLength = config.options().get<float>("tfLength");
  const auto seed = config.options().get<int>("seed");
  const auto amplitude = config.options().get<float>("amplitude");
  const auto noise = config.options().get<float>("noise");
  const auto inputFile = config.options().get<std::string>("input-file");
  const auto inputEntry = static_cast<long long>(config.options().get<int>("input-entry"));

  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));

  WorkflowSpec workflow;
  workflow.emplace_back(generateCMVsCRU(crus, delayCRUs, timeframes, delay, delayTime, delayEveryN, delayTimeCRUs, dropTFsRandom, rangeTFsDrop, tfLength, amplitude, noise, seed, inputFile, inputEntry));

  auto& hbfu = o2::raw::HBFUtils::Instance();
  long startTime = hbfu.startTime > 0 ? hbfu.startTime : std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  o2::conf::ConfigurableParam::updateFromString(fmt::format("HBFUtils.startTime={}", startTime).data());
  o2::conf::ConfigurableParam::updateFromString(fmt::format("HBFUtils.nHBFPerTF={}", hbfu.nHBFPerTF).data());
  o2::raw::HBFUtilsInitializer hbfIni(config, workflow);

  return workflow;
}
