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

#include <string>
#include <fairlogger/Logger.h>
#include "Framework/InputRecordWalker.h"
#include "Framework/DataRefUtils.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsRaw/RDHUtils.h"
#include "CTPWorkflowLumi/RawDecoderSpec.h"
#include "CommonUtils/VerbosityConfig.h"
#include "Framework/InputRecord.h"
#include "DataFormatsCTP/TriggerOffsetsParam.h"
#include "Framework/CCDBParamSpec.h"
#include "DataFormatsCTP/Configuration.h"
#include "CommonConstants/LHCConstants.h"
#include <DataFormatsParameters/GRPLHCIFData.h>
#include <filesystem>

using namespace o2::ctp::reco_workflow;

void RawDecoderSpec::init(framework::InitContext& ctx)
{
  mCheckConsistency = ctx.options().get<bool>("check-consistency");
  mDecoder.setCheckConsistency(mCheckConsistency);
  mDecodeinputs = ctx.options().get<bool>("ctpinputs-decoding");
  mDecoder.setDecodeInps(mDecodeinputs);
  mNTFToIntegrate = ctx.options().get<int>("ntf-to-average");
  LOG(info) << "Window size: " << mNTFToIntegrate << " TFs";
  mVerbose = ctx.options().get<bool>("use-verbose-mode");
  int maxerrors = ctx.options().get<int>("print-errors-num");
  mDecoder.setVerbose(mVerbose);
  mDecoder.setDoLumi(mDoLumi);
  mDecoder.setDoDigits(mDoDigits);
  mDecoder.setMAXErrors(maxerrors);
  std::string lumiinp1 = ctx.options().get<std::string>("lumi-inp1");
  std::string lumiinp2 = ctx.options().get<std::string>("lumi-inp2");
  int inp1 = mDecoder.setLumiInp(1, lumiinp1);
  int inp2 = mDecoder.setLumiInp(2, lumiinp2);
  mOutputLumiInfo.inp1 = inp1;
  mOutputLumiInfo.inp2 = inp2;
  mMaxInputSize = ctx.options().get<int>("max-input-size");
  mMaxInputSizeFatal = ctx.options().get<bool>("max-input-size-fatal");
  LOG(info) << "CTP reco init done. Inputs decoding here:" << mDecodeinputs << " DoLumi:" << mDoLumi << " DoDigits:" << mDoDigits << " NTF:" << mNTFToIntegrate << " Lumi inputs:" << lumiinp1 << ":" << inp1 << " " << lumiinp2 << ":" << inp2 << " Max errors:" << maxerrors << " Max input size:" << mMaxInputSize << " MaxInputSizeFatal:" << mMaxInputSizeFatal << " CheckConsistency:" << mCheckConsistency;
  mMassiOutDir = ctx.options().get<std::string>("massi-out-dir");
  LOG(info) << "Massi output dir:" << mMassiOutDir;
  mCrossSection = ctx.options().get<double>("cross-section");
  LOG(info) << "Cross section (ub): " << mCrossSection;
  mReorderDepth = ctx.options().get<int>("tf-reorder-depth");
  // mOutputLumiInfo.printInputs();
}
void RawDecoderSpec::endOfStream(framework::EndOfStreamContext& ec)
{
  auto clsEA = mDecoder.getClassErrorsA();
  auto clsEB = mDecoder.getClassErrorsB();
  auto cntCA = mDecoder.getClassCountersA();
  auto cntCB = mDecoder.getClassCountersB();
  int totClasses = 0;
  for (int i = 0; i < o2::ctp::CTP_NCLASSES; i++) {
    mClsEA[i] += clsEA[i];
    mClsEB[i] += clsEB[i];
    mClsA[i] += cntCA[i];
    mClsB[i] += cntCB[i];
    totClasses += cntCA[i];
  }
  auto& TFOrbits = mDecoder.getTFOrbits();
  std::sort(TFOrbits.begin(), TFOrbits.end());
  size_t l = TFOrbits.size();
  uint32_t o0 = 0;
  if (l) {
    o0 = TFOrbits[0];
  }
  int nmiss = 0;
  int nprt = 0;
  std::cout << "Missing orbits:";
  for (int i = 1; i < l; i++) {
    if ((TFOrbits[i] - o0) > 0x20) {
      if (nprt < 20) {
        std::cout << " " << o0 << "-" << TFOrbits[i];
      }
      nmiss += (TFOrbits[i] - o0) / 0x20;
      nprt++;
    }
    o0 = TFOrbits[i];
  }
  std::cout << std::endl;
  LOG(info) << "Number of non continous TF:" << nmiss << std::endl;
  LOG(info) << "Lost in shiftInputs:" << mLostDueToShiftInps;
  LOG(info) << "Lost in addDigit Inputs:" << mIRRejected << " Classes:" << mTCRRejected;
  if (mErrorIR || mErrorTCR) {
    LOG(error) << "# of IR errors:" << mErrorIR << " TCR errors:" << mErrorTCR << std::endl;
  }
  if (mCheckConsistency) {
    LOG(info) << "Lost due to the shift Consistency Checker:" << mDecoder.getLostDueToShiftCls();
    LOG(info) << "Total classes:" << totClasses;
    auto ctpcfg = mDecoder.getCTPConfig();
    for (int i = 0; i < o2::ctp::CTP_NCLASSES; i++) {
      std::string name = ctpcfg.getClassNameFromIndex(i);
      if (mClsEA[i]) {
        LOG(error) << " Class without inputs:";
      }
      LOG(important) << "CLASS:" << name << ":" << i << " Cls=>Inp:" << mClsA[i] << " Inp=>Cls:" << mClsB[i] << "  ErrorsCls=>Inps:" << mClsEA[i] << "  MissingInps=>Cls:" << mClsEB[i];
    }
  }
  flushAllPendingTFs();
  if (mTFsInCurrentWindow > 0) {
    double timeInterval = orbitTime * mOrbitsInCurrentWindow;
    double totalLumi1 = 0.0;
    double totalLumi2 = 0.0;
    double totalLumiErr1 = 0.0;
    double totalLumiErr2 = 0.0;
    size_t filledBCs = mLHCBCs.count();
    for (size_t bc = 0; bc < mCountsPerBC1.size(); ++bc) {
      if (mCountsPerBC1[bc] > 0) {
        double rate1 = mCountsPerBC1[bc] / timeInterval;
        double lumi1 = rate1 / mCrossSection;
        double lumiErr1 = std::sqrt(mCountsPerBC1[bc]) / (timeInterval * mCrossSection);
        auto [mu, correctedRate1] = pileupCorrection(rate1);
        double correctedLumi1 = correctedRate1 / mCrossSection;
        writeMassiLinePerBC(bc, mWindowStartTime, lumi1, lumiErr1, correctedLumi1, correctedRate1, mu);
      }
      if (mLHCBCs.test(bc)) {
        totalLumi1 += mCountsPerBC1[bc] / (timeInterval * mCrossSection);
        totalLumi2 += mCountsPerBC2[bc] / (timeInterval * mCrossSection);
        totalLumiErr1 += std::sqrt(mCountsPerBC1[bc]) / (timeInterval * mCrossSection);
        totalLumiErr2 += std::sqrt(mCountsPerBC2[bc]) / (timeInterval * mCrossSection);
        writeMassiLineLumi(mWindowStartTime, totalLumi1, totalLumiErr1);
      }
    }
    LOG(info) << "Flushed trailing partial window of " << mTFsInCurrentWindow << " TFs at end of stream";
  }
  // Calculate and print total luminosity for given fill
  double totalFillCountsInp1 = 0.0;
  double totalFillCountsInp2 = 0.0;
  for (const auto& count : mTotalCountsPerBC1) {
    totalFillCountsInp1 += count;
  }
  for (const auto& count : mTotalCountsPerBC2) {
    totalFillCountsInp2 += count;
  }
  // Estimate the total integrated luminosity for the fill in ub^-1 and the rate in Hz
  double avgRate1 = totalFillCountsInp1 / mTotalElapsedTime;
  double fillDurationSec = (mRunInfo.eor - mRunInfo.sor) / 1000.0;
  double totalIntLumiInp1 = totalFillCountsInp1 / mCrossSection;
  double estimatedTotalIntLumiInp1 = (avgRate1 / mCrossSection) * fillDurationSec;
  LOG(info) << "Total Integrated Luminosity Input 1: " << totalIntLumiInp1 << " ub^-1" << " Rate (vis): " << avgRate1 << " Hz, Estimated Total Integrated Lumi: " << estimatedTotalIntLumiInp1 << " ub^-1";
  // Close files at end of stream
  for (auto& [bucket, ofs] : mMassiFiles) {
    ofs.close();
  }
}
void RawDecoderSpec::run(framework::ProcessingContext& ctx)
{
  updateTimeDependentParams(ctx);
  mOutputDigits.clear();
  std::map<o2::InteractionRecord, CTPDigit> digits;
  using InputSpec = o2::framework::InputSpec;
  using ConcreteDataTypeMatcher = o2::framework::ConcreteDataTypeMatcher;
  using Lifetime = o2::framework::Lifetime;
  // setUpDummyLink
  auto& inputs = ctx.inputs();
  auto dummyOutput = [&ctx, this]() {
    if (this->mDoDigits) {
      ctx.outputs().snapshot(o2::framework::Output{"CTP", "DIGITS", 0}, this->mOutputDigits);
    }
    if (this->mDoLumi) {
      ctx.outputs().snapshot(o2::framework::Output{"CTP", "LUMI", 0}, this->mOutputLumiInfo);
    }
  };
  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  {
    static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
    std::vector<InputSpec> dummy{InputSpec{"dummy", o2::framework::ConcreteDataMatcher{"CTP", "RAWDATA", 0xDEADBEEF}}};
    for (const auto& ref : o2::framework::InputRecordWalker(inputs, dummy)) {
      const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadSize = o2::framework::DataRefUtils::getPayloadSize(ref);
      if (payloadSize == 0) {
        auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
        if (++contDeadBeef <= maxWarn) {
          LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
               dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadSize,
               contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
        }
        dummyOutput();
        return;
      }
    }
    contDeadBeef = 0; // if good data, reset the counter
  }
  //
  std::vector<LumiInfo> lumiPointsHBF1;
  std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{"CTP", "RAWDATA"}, Lifetime::Timeframe}};
  bool fatal_flag = 0;
  size_t payloadSize = 0;
  bool gotFirstOrbit = false;

  for (const auto& ref : o2::framework::InputRecordWalker(inputs, filter)) {
    const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    if (!gotFirstOrbit) {
      mFirstOrbit = dh->firstTForbit;
      gotFirstOrbit = true;
      if (mHavePrevTF) {
        uint32_t expectedOrbit = mPrevTFLastOrbit;
        if (mFirstOrbit != expectedOrbit) {
          int64_t diff = static_cast<int64_t>(mFirstOrbit) - static_cast<int64_t>(mPrevTFLastOrbit);
          if (diff < 0) {
            LOG(warning) << "TF arrived out of order: previous TF ended at orbit " << mPrevTFLastOrbit << ", this TF starts at " << mFirstOrbit << " (orbit went backwards by " << diff << ")";
          } else if (diff > 0) {
            LOG(warning) << "Gap detected: previous TF ended at orbit " << expectedOrbit << ", this TF starts at " << mFirstOrbit << " (missing " << (mFirstOrbit - expectedOrbit) << " orbits)";
          }
        }
      }
    }
    mPrevTFLastOrbit = mFirstOrbit + mRunInfo.orbitsPerTF;
    mHavePrevTF = true;
    if (mMaxInputSize > 0) {
      payloadSize += o2::framework::DataRefUtils::getPayloadSize(ref);
    }
  }
  LOG(info) << "mFirstOrbit for this TF: " << mFirstOrbit << " gotFirstOrbit: " << gotFirstOrbit;
  // if (payloadSize > (size_t)mMaxInputSize) {
  if (mMaxInputSize > 0 && payloadSize > (size_t)mMaxInputSize) {
    if (mMaxInputSizeFatal) {
      fatal_flag = 1;
      LOG(error) << "Input data size bigger than threshold: " << mMaxInputSize << " < " << payloadSize << " decoding TF and exiting.";
      // LOG(fatal) << "Input data size:" << payloadSize; - fatal issued in decoder
    } else {
      LOG(error) << "Input data size:" << payloadSize << " sending dummy output";
      dummyOutput();
      return;
    }
  }

  int ret = 0;
  if (fatal_flag) {
    ret = mDecoder.decodeRawFatal(inputs, filter);
  } else {
    ret = mDecoder.decodeRaw(inputs, filter, mOutputDigits, lumiPointsHBF1);
  }
  if (ret == 1) {
    dummyOutput();
    return;
  }
  if (mDoDigits) {
    LOG(info) << "[CTPRawToDigitConverter - run] Writing " << mOutputDigits.size() << " digits. IR rejected:" << mDecoder.getIRRejected() << " TCR rejected:" << mDecoder.getTCRRejected();
    ctx.outputs().snapshot(o2::framework::Output{"CTP", "DIGITS", 0}, mOutputDigits);
    mLostDueToShiftInps += mDecoder.getLostDueToShiftInp();
    mErrorIR += mDecoder.getErrorIR();
    mErrorTCR += mDecoder.getErrorTCR();
    mIRRejected += mDecoder.getIRRejected();
    mTCRRejected += mDecoder.getTCRRejected();
    // Luminosity per bunch crossing
    computeLumiPerBC(mOutputDigits, mFirstOrbit, static_cast<uint32_t>(mRunInfo.orbitsPerTF));
  }
  if (mDoLumi) {
    uint32_t tfCountsT = 0;
    uint32_t tfCountsV = 0;
    for (auto const& lp : lumiPointsHBF1) {
      tfCountsT += lp.counts;
      tfCountsV += lp.countsFV0;
    }
    // LOG(info) << "Lumi rate:" << tfCounts/(128.*88e-6);
    // FT0
    mHistoryT.push_back(tfCountsT);
    mCountsT += tfCountsT;
    if (mHistoryT.size() <= mNTFToIntegrate) {
      mNHBIntegratedT += lumiPointsHBF1.size();
    } else {
      mCountsT -= mHistoryT.front();
      mHistoryT.pop_front();
    }
    // FV0
    mHistoryV.push_back(tfCountsV);
    mCountsV += tfCountsV;
    if (mHistoryV.size() <= mNTFToIntegrate) {
      mNHBIntegratedV += lumiPointsHBF1.size();
    } else {
      mCountsV -= mHistoryV.front();
      mHistoryV.pop_front();
    }
    //
    if (mNHBIntegratedT || mNHBIntegratedV) {
      mOutputLumiInfo.orbit = lumiPointsHBF1[0].orbit;
    }
    mOutputLumiInfo.counts = mCountsT;

    mOutputLumiInfo.countsFV0 = mCountsV;
    mOutputLumiInfo.nHBFCounted = mNHBIntegratedT;
    mOutputLumiInfo.nHBFCountedFV0 = mNHBIntegratedV;
    if (mVerbose) {
      mOutputLumiInfo.printInputs();
      LOGP(info, "Orbit {}: {}/{} counts inp1/inp2 in {}/{} HBFs -> lumi_inp1 = {:.3e}+-{:.3e} lumi_inp2 = {:.3e}+-{:.3e}", mOutputLumiInfo.orbit, mCountsT, mCountsV, mNHBIntegratedT, mNHBIntegratedV, mOutputLumiInfo.getLumi(), mOutputLumiInfo.getLumiError(), mOutputLumiInfo.getLumiFV0(), mOutputLumiInfo.getLumiFV0Error());
    }
    ctx.outputs().snapshot(o2::framework::Output{"CTP", "LUMI", 0}, mOutputLumiInfo);
  }
}
// Function to compute luminosity per BC from the interaction counts from CTP digits
// std::pair<std::array<double, o2::constants::lhc::LHCMaxBunches>, std::array<double, o2::constants::lhc::LHCMaxBunches>>
void RawDecoderSpec::computeLumiPerBC(const o2::pmr::vector<CTPDigit>& ctpdigits, uint32_t firstOrbit, uint32_t orbitsPerTF)
{
  int inp1 = mOutputLumiInfo.inp1;
  int inp2 = mOutputLumiInfo.inp2;

  uint64_t inputMask1 = 1ull << (inp1 - 1); // TVX
  uint64_t inputMask2 = 1ull << (inp2 - 1); // VBA

  std::array<double, o2::constants::lhc::LHCMaxBunches> tfCountsPerBC1{};
  std::array<double, o2::constants::lhc::LHCMaxBunches> tfCountsPerBC2{};

  for (const auto& digit : ctpdigits) {
    uint32_t orbit = digit.intRecord.orbit;
    if (orbit < firstOrbit || orbit >= firstOrbit + orbitsPerTF) {
      LOG(warning) << "Digit orbit " << orbit << " outside expected TF range [" << firstOrbit << ", " << (firstOrbit + orbitsPerTF) << ") - skipping";
      continue;
    }
    uint64_t mask = digit.CTPInputMask.to_ullong();
    uint16_t bc = digit.intRecord.bc;
    if (bc < o2::constants::lhc::LHCMaxBunches) {
      if (mask & inputMask1) {
        tfCountsPerBC1[bc] += 1.0;
      }  
      if (mask & inputMask2) {
        tfCountsPerBC2[bc] += 1.0;
      }
    }
  }
  int64_t unixTimeStart = unixTimeForOrbitStart(firstOrbit);
  if (mPendingTFs.count(firstOrbit)) {
    LOG(warning) << "Duplicate firstOrbit " << firstOrbit << " received - overwriting pending entry";
  }
  mPendingTFs[firstOrbit] = PendingTF{tfCountsPerBC1, tfCountsPerBC2, unixTimeStart, orbitsPerTF};
  if (!mPendingTFs.empty()) {
    uint32_t smallestPending = mPendingTFs.begin()->first;
    if (firstOrbit < smallestPending) {
      LOG(warning) << "Late TF: firstOrbit=" << firstOrbit << " arrived after smallest pending=" << smallestPending;
    }
  }
  flushReadyTFs();
  // integrateLumi(tfCountsPerBC1, tfCountsPerBC2, unixTimeStart, orbitsPerTF);
}
// Accumulate luminosity per BC over multiple time frames
void RawDecoderSpec::integrateLumi(const std::array<double, o2::constants::lhc::LHCMaxBunches>& tfCounts1, const std::array<double, o2::constants::lhc::LHCMaxBunches>& tfCounts2, int64_t unixTimeStart, uint32_t nOrbitsThisTF)
{
  if (mTFsInCurrentWindow == 0) {
    mWindowStartTime = unixTimeStart;
  }

  for (size_t bc = 0; bc < mCountsPerBC1.size(); ++bc) {
    mCountsPerBC1[bc] += tfCounts1[bc];
    mTotalCountsPerBC1[bc] += tfCounts1[bc];
  }
  for (size_t bc = 0; bc < mCountsPerBC2.size(); ++bc) {
    mCountsPerBC2[bc] += tfCounts2[bc];
    mTotalCountsPerBC2[bc] += tfCounts2[bc];
  }
  mTotalElapsedTime += nOrbitsThisTF * orbitTime;
  mOrbitsInCurrentWindow += nOrbitsThisTF;
  ++mTFsInCurrentWindow;

  if (mTFsInCurrentWindow < mNTFToIntegrate) {
    return; // Window not yet filled
  }

  if (mTFsInCurrentWindow >= mNTFToIntegrate) {
    double timeInterval = orbitTime * mOrbitsInCurrentWindow; // Total time in seconds for the current window
    // Count number of filled BCs
    size_t filledBCs = mLHCBCs.count();
    // Total lumi over filled BCs for this window
    double totalLumi1 = 0.0;
    double totalLumi2 = 0.0;
    double totalLumiErr1 = 0.0;
    double totalLumiErr2 = 0.0;
    for (size_t bc = 0; bc < mCountsPerBC1.size(); ++bc) { // Luminosity per BC
      if (mCountsPerBC1[bc] > 0 || mCountsPerBC2[bc] > 0) {
        double rate1 = mCountsPerBC1[bc] / timeInterval;
        double rate2 = mCountsPerBC2[bc] / timeInterval;
        double lumi1 = rate1 / mCrossSection;
        double lumi2 = rate2 / mCrossSection;
        double lumiErr1 = std::sqrt(mCountsPerBC1[bc]) / (timeInterval * mCrossSection);
        double lumiErr2 = std::sqrt(mCountsPerBC2[bc]) / (timeInterval * mCrossSection);
        auto [mu, correctedRate1] = pileupCorrection(rate1);
        double correctedLumi1 = correctedRate1 / mCrossSection;
        if (mCountsPerBC1[bc] > 0) {
          //  LOG(info) << "BC: " << bc + 1 << " Rate: " << rate1 << " Corrected Rate: " << correctedRate1 << " mu: " << mu;
          writeMassiLinePerBC(bc, mWindowStartTime, lumi1, lumiErr1, correctedRate1, correctedLumi1, mu);
        }
      }

      // Total luminosity over filled BCs for this window
      if (mLHCBCs.test(bc)) {
        totalLumi1 += mCountsPerBC1[bc] / (timeInterval * mCrossSection);
        totalLumi2 += mCountsPerBC2[bc] / (timeInterval * mCrossSection);
        totalLumiErr1 += std::sqrt(mCountsPerBC1[bc]) / (timeInterval * mCrossSection);
        totalLumiErr2 += std::sqrt(mCountsPerBC2[bc]) / (timeInterval * mCrossSection);
      }
    }
    writeMassiLineLumi(mWindowStartTime, totalLumi1, totalLumiErr1);
    // Reset counters for the next window
    mCountsPerBC1.fill(0.0);
    mCountsPerBC2.fill(0.0);
    mTFsInCurrentWindow = 0;
    mOrbitsInCurrentWindow = 0;
  }
}
void RawDecoderSpec::writeMassiLinePerBC(int bc, int64_t unixTimeStart, double lumi, double lumiErr, double correctedRate, double correctedLumi, double mu)
{
  int rfBucket = (bc * 10) + 1;
  auto it = mMassiFiles.find(rfBucket);
  if (it == mMassiFiles.end()) {
    std::string dirPath = mMassiOutDir + "/" + std::to_string(mMassiYear) + "/lumi/" + mFillNumber;

    std::error_code ec;
    std::filesystem::create_directories(dirPath, ec);
    if (ec) {
      LOG(error) << "Failed to create Massi output directory " << dirPath << ": " << ec.message();
      return;
    }
    std::string filename = dirPath + "/" + mFillNumber + "_lumi_" + std::to_string(rfBucket) + "_ALICE.txt";
    auto result = mMassiFiles.emplace(rfBucket, std::ofstream(filename, std::ios::app));
    it = result.first;
  }
  std::ofstream& ofs = it->second;
  ofs << std::fixed << std::setprecision(0) << unixTimeStart << " " << mStableBeams << " ";
  ofs << (std::abs(lumi) < 1e-3 ? std::scientific : std::fixed) << std::setprecision(7) << lumi << " ";
  ofs << (std::abs(lumiErr) < 1e-3 ? std::scientific : std::fixed) << std::setprecision(7) << lumiErr << " ";
  ofs << (std::abs(correctedLumi) < 1e-3 ? std::scientific : std::fixed) << std::setprecision(7) << correctedLumi << " ";
  ofs << std::fixed << std::setprecision(7) << correctedRate << " " << mu << " " << std::endl;
  ofs.flush();
}
void RawDecoderSpec::writeMassiLineLumi(int64_t unixTimeStart, double lumi, double lumiErr)
{
  std::string dirPath = mMassiOutDir + "/" + std::to_string(mMassiYear) + "/lumi/" + mFillNumber;
  std::error_code ec;
  std::filesystem::create_directories(dirPath, ec);
  if (ec) {
    LOG(error) << "Failed to create Massi output directory " << dirPath << ": " << ec.message();
    return;
  }
  std::string filename = dirPath + "/" + mFillNumber + "_lumi_ALICE.txt";
  std::ofstream ofs(filename, std::ios::app);
  ofs << std::fixed << std::setprecision(0) << unixTimeStart << " " << mStableBeams << " ";
  ofs << (std::abs(lumi) < 1e-3 ? std::scientific : std::fixed) << std::setprecision(7) << lumi << " ";
  ofs << (std::abs(lumiErr) < 1e-3 ? std::scientific : std::fixed) << std::setprecision(7) << lumiErr << " " << std::endl;
  ofs.flush();
}
int64_t RawDecoderSpec::unixTimeForOrbitStart(uint32_t orbit) const
{
  int64_t orbitResetTimeMUS = mRunInfo.orbitReset;
  return (orbitResetTimeMUS + static_cast<int64_t>(orbit) * o2::constants::lhc::LHCOrbitMUS) * 1e-3; // Return in milliseconds
}
int RawDecoderSpec::yearFromUnixTime(int64_t unixTimeStart) const
{
  std::time_t time = static_cast<std::time_t>(unixTimeStart);
  std::tm* tm = std::gmtime(&time);
  return tm->tm_year + 1900;
}
void RawDecoderSpec::fetchRunInfo(int runNumber)
{
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  mRunInfo = o2::parameters::AggregatedRunInfo::buildAggregatedRunInfo_DATA(ccdbMgr, runNumber);
  mOrbitsPerTF = mRunInfo.orbitsPerTF;
  mMassiYear = yearFromUnixTime(mRunInfo.sor / 1000.0);
  mOrbitResetTimeSec = mRunInfo.orbitReset * 1e-6;
  mRunStartTime = mRunInfo.sor / 1000;
  mRunEndTime = mRunInfo.eor / 1000;
  LOG(info) << "Run start time: " << mRunStartTime << " Run end time: " << mRunEndTime;
}
void RawDecoderSpec::flushReadyTFs()
{
  while (mPendingTFs.size() > mReorderDepth) {
    auto it = mPendingTFs.begin();
    integrateLumi(it->second.countsPerBC1, it->second.countsPerBC2, it->second.unixTimeStart, it->second.nOrbitsThisTF);
    mPendingTFs.erase(it);
  }
}
void RawDecoderSpec::flushAllPendingTFs()
{
  while (!mPendingTFs.empty()) {
    auto it = mPendingTFs.begin();
    integrateLumi(it->second.countsPerBC1, it->second.countsPerBC2, it->second.unixTimeStart, it->second.nOrbitsThisTF);
    mPendingTFs.erase(it);
  }
}
std::pair<double, double> RawDecoderSpec::pileupCorrection(double rate) const
{
  double p = rate / o2::constants::lhc::LHCRevFreq;
  if (p >= 1.0) {
    LOG(warning) << "Pile-up correction: p = " << p << " >= 1";
    return {0, 0};
  }
  double mu = -std::log(1 - p);
  double correctedRate = mu * o2::constants::lhc::LHCRevFreq;
  return {mu, correctedRate};
}
o2::framework::DataProcessorSpec o2::ctp::reco_workflow::getRawDecoderSpec(bool askDISTSTF, bool digits, bool lumi)
{
  if (!digits && !lumi) {
    throw std::runtime_error("all outputs were disabled");
  }
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("TF", o2::framework::ConcreteDataTypeMatcher{"CTP", "RAWDATA"}, o2::framework::Lifetime::Timeframe);
  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }

  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("ctpconfig", "CTP", "CTPCONFIG", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("CTP/Config/Config", 1));
  inputs.emplace_back("grplhcif", "GLO", "GRPLHCIF", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("GLO/Config/GRPLHCIF"));
  inputs.emplace_back("trigoffset", "CTP", "Trig_Offset", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("CTP/Config/TriggerOffsets"));
  if (digits) {
    outputs.emplace_back("CTP", "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  }
  if (lumi) {
    outputs.emplace_back("CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe);
  }
  return o2::framework::DataProcessorSpec{
    "ctp-raw-decoder-lumi",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<o2::ctp::reco_workflow::RawDecoderSpec>(digits, lumi)},
    o2::framework::Options{
      {"ntf-to-average", o2::framework::VariantType::Int, 100, {"Time interval for averaging luminosity in units of TF"}},
      {"print-errors-num", o2::framework::VariantType::Int, 3, {"Max number of errors to print"}},
      {"lumi-inp1", o2::framework::VariantType::String, "TVX", {"The first input used for online lumi. Name in capital."}},
      {"lumi-inp2", o2::framework::VariantType::String, "VBA", {"The second input used for online lumi. Name in capital."}},
      {"use-verbose-mode", o2::framework::VariantType::Bool, false, {"Verbose logging"}},
      {"max-input-size", o2::framework::VariantType::Int, 0, {"Do not process input if bigger than max size, 0 - do not check"}},
      {"max-input-size-fatal", o2::framework::VariantType::Bool, false, {"If true issue fatal error otherwise error only"}},
      {"check-consistency", o2::framework::VariantType::Bool, false, {"If true checks digits consistency using ctp config"}},
      {"ctpinputs-decoding", o2::framework::VariantType::Bool, false, {"Inputs alignment: true - raw decoder - has to be compatible with CTF decoder: allowed options: 10,01,00"}},
      {"cross-section", o2::framework::VariantType::Double, 59500.0, {"Cross-section in ub, default for pp collisions"}},
      {"tf-reorder-depth", o2::framework::VariantType::Int, 300, {"Number of TFs to buffer to correct out of-order TF delivery"}},
      {"massi-out-dir", o2::framework::VariantType::String, ".", {"Output directory for Massi files"}}}};
}
void RawDecoderSpec::updateTimeDependentParams(framework::ProcessingContext& pc)
{
  if (pc.services().get<o2::framework::TimingInfo>().globalRunNumberChanged) {
    pc.inputs().get<o2::ctp::TriggerOffsetsParam*>("trigoffset");
    const auto& trigOffsParam = o2::ctp::TriggerOffsetsParam::Instance();
    LOG(info) << "updateing TroggerOffsetsParam: inputs L0_L1:" << trigOffsParam.L0_L1 << " classes L0_L1:" << trigOffsParam.L0_L1_classes;
    const auto ctpcfg = pc.inputs().get<o2::ctp::CTPConfiguration*>("ctpconfig");
    if (ctpcfg != nullptr) {
      mDecoder.setCTPConfig(*ctpcfg);
      LOG(info) << "ctpconfig for run done:" << mDecoder.getCTPConfig().getRunNumber();
    }
    const auto grplhcif = pc.inputs().get<o2::parameters::GRPLHCIFData*>("grplhcif");
    if (grplhcif != nullptr) {
      LOG(info) << "GRPLHCIF injection scheme: " << grplhcif->getInjectionScheme();
      LOG(info) << "Bunch filling with time: " << grplhcif->getBunchFillingTime();
      LOG(info) << "Fill number time: " << grplhcif->getFillNumberTime();
      LOG(info) << "Injection scheme time: " << grplhcif->getInjectionSchemeTime();

      // Get filled bunches
      auto bfilling = grplhcif->getBunchFilling();
      std::vector<int> bcs = bfilling.getFilledBCs();
      LOG(info) << "Filled BCs: " << bcs.size();
      mLHCBCs.reset();
      for (auto const& bc : bcs) {
        mLHCBCs.set(bc, 1);
      }
      mFillNumber = std::to_string(grplhcif->getFillNumber());
    }
    int runNumber = pc.services().get<o2::framework::TimingInfo>().runNumber;
    fetchRunInfo(runNumber);
  }
}
