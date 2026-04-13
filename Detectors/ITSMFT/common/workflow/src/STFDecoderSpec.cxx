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

/// \file   STFDecoderSpec.cxx
/// \brief  Device to decode ITS or MFT raw data from STF
/// \author ruben.shahoyan@cern.ch

#include <vector>

#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DeviceSpec.h"
#include "Framework/CCDBParamSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTReconstruction/RawPixelDecoder.h"
#include "ITSMFTReconstruction/DigitPixelReader.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "ITSMFTReconstruction/GBTLink.h"
#include "ITSMFTWorkflow/STFDecoderSpec.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/VerbosityConfig.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DataFormatsParameters/GRPECSObject.h"

namespace o2
{
namespace itsmft
{

using namespace o2::framework;

///_______________________________________
template <class Mapping>
STFDecoder<Mapping>::STFDecoder(const STFDecoderInp& inp, std::shared_ptr<o2::base::GRPGeomRequest> gr)
  : mDoClusters(inp.doClusters), mDoPatterns(inp.doPatterns), mDoDigits(inp.doDigits), mDoCalibData(inp.doCalib), mDoStaggering(inp.doStaggering), mAllowReporting(inp.allowReporting), mVerifyDecoder(inp.verifyDecoder), mInputSpec(inp.inputSpec), mGGCCDBRequest(gr)
{
  mSelfName = o2::utils::Str::concat_string(Mapping::getName(), "STFDecoder");
  mTimer.Stop();
  mTimer.Reset();
  if (mDoStaggering) {
    mLayers = Mapping::NLayers;
    mEstNDig.resize(mLayers, 0);
    mEstNClus.resize(mLayers, 0);
    mEstNClusPatt.resize(mLayers, 0);
    mEstNCalib.resize(mLayers, 0);
  }
}

///_______________________________________
template <class Mapping>
void STFDecoder<Mapping>::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  try {
    auto v0 = o2::utils::Str::tokenize(mInputSpec, ':');
    auto v1 = o2::utils::Str::tokenize(v0[1], '/');
    auto v2 = o2::utils::Str::tokenize(v1[1], '?');
    header::DataOrigin dataOrig;
    header::DataDescription dataDesc;
    dataOrig.runtimeInit(v1[0].c_str());
    dataDesc.runtimeInit(v2[0].c_str());
    for (int iLayer{0}; iLayer < mLayers; ++iLayer) {
      auto& dec = mDecoder.emplace_back(std::make_unique<RawPixelDecoder<Mapping>>());
      dec->setUserDataOrigin(dataOrig);
      dec->setUserDataDescription(dataDesc);
      dec->init(); // is this no-op?
    }
  } catch (const std::exception& e) {
    LOG(error) << "exception was thrown in decoder creation: " << e.what();
    throw;
  } catch (...) {
    LOG(error) << "non-std::exception was thrown in decoder creation";
    throw;
  }
  mApplyNoiseMap = !ic.options().get<bool>("ignore-noise-map");
  mUseClusterDictionary = !ic.options().get<bool>("ignore-cluster-dictionary");
  try {
    float fr = ic.options().get<float>("rof-length-error-freq");
    mROFErrRepIntervalMS = fr <= 0. ? -1 : long(fr * 1e3);
    mNThreads = std::max(1, ic.options().get<int>("nthreads"));
    mUnmutExtraLanes = ic.options().get<bool>("unmute-extra-lanes");
    mVerbosity = ic.options().get<int>("decoder-verbosity");
    auto dmpSz = ic.options().get<int>("stop-raw-data-dumps-after-size");
    if (dmpSz > 0) {
      mMaxRawDumpsSize = size_t(dmpSz) * 1024 * 1024;
    }
    mDumpOnError = ic.options().get<int>("raw-data-dumps");
    if (mDumpOnError < 0) {
      mDumpOnError = -mDumpOnError;
      mDumpFrom1stPipeline = true;
    }
    if (mDumpOnError >= int(GBTLink::RawDataDumps::DUMP_NTYPES)) {
      throw std::runtime_error(fmt::format("unknown raw data dump level {} requested", mDumpOnError));
    }
    auto dumpDir = ic.options().get<std::string>("raw-data-dumps-directory");
    if (mDumpOnError != int(GBTLink::RawDataDumps::DUMP_NONE) && (!dumpDir.empty() && !o2::utils::Str::pathIsDirectory(dumpDir))) {
      throw std::runtime_error(fmt::format("directory {} for raw data dumps does not exist", dumpDir));
    }
    for (int iLayer{0}; iLayer < mLayers; ++iLayer) {
      mDecoder[iLayer]->setNThreads(mNThreads);
      mDecoder[iLayer]->setAlwaysParseTrigger(ic.options().get<bool>("always-parse-trigger"));
      mDecoder[iLayer]->setAllowEmptyROFs(ic.options().get<bool>("allow-empty-rofs"));
      mDecoder[iLayer]->setRawDumpDirectory(dumpDir);
      mDecoder[iLayer]->setFillCalibData(mDoCalibData);
      mDecoder[iLayer]->setVerifyDecoder(mVerifyDecoder);
      bool ignoreRampUp = !ic.options().get<bool>("accept-rof-rampup-data");
      mDecoder[iLayer]->setSkipRampUpData(ignoreRampUp);
    }
  } catch (const std::exception& e) {
    LOG(error) << "exception was thrown in decoder configuration: " << e.what();
    throw;
  } catch (...) {
    LOG(error) << "non-std::exception was thrown in decoder configuration";
    throw;
  }

  if (mDoClusters) {
    mClusterer = std::make_unique<Clusterer>();
    mClusterer->setNChips(Mapping::getNChips());
  }

  if (mDoStaggering) {
    Mapping map;
    for (uint32_t iLayer{0}; iLayer < mLayers; ++iLayer) {
      std::vector<o2::framework::InputSpec> filter;
      for (const auto feeID : map.getLayer2FEEIDs(iLayer)) {
        filter.emplace_back("filter", ConcreteDataMatcher{Mapping::getOrigin(), o2::header::gDataDescriptionRawData, (o2::header::DataHeader::SubSpecificationType)feeID});
      }
      mDecoder[iLayer]->setInputFilter(filter);
    }
  }
}

///_______________________________________
template <class Mapping>
void STFDecoder<Mapping>::run(ProcessingContext& pc)
{
  updateTimeDependentParams(pc);
  static bool firstCall = true;
  if (!firstCall && pc.services().get<o2::framework::TimingInfo>().globalRunNumberChanged) { // reset at the beginning of the new run
    reset();
  }
  if (firstCall) {
    firstCall = false;
    for (int iLayer{0}; iLayer < mLayers; ++iLayer) {
      mDecoder[iLayer]->setInstanceID(pc.services().get<const o2::framework::DeviceSpec>().inputTimesliceId);
      mDecoder[iLayer]->setNInstances(pc.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices);
      mDecoder[iLayer]->setVerbosity(mDecoder[iLayer]->getInstanceID() == 0 ? mVerbosity : (mUnmutExtraLanes ? mVerbosity : -1));
    }
    mAllowReporting &= (mDecoder[0]->getInstanceID() == 0) || mUnmutExtraLanes;
  }

  int nSlots = pc.inputs().getNofParts(0);
  double timeCPU0 = mTimer.CpuTime(), timeReal0 = mTimer.RealTime();
  mTimer.Start(false);
  auto orig = Mapping::getOrigin();

  // these are accumulated from each layer
  auto& chipStatus = pc.outputs().make<std::vector<char>>(Output{orig, "CHIPSSTATUS", 0}, (size_t)Mapping::getNChips());
  auto& linkErrors = pc.outputs().make<std::vector<GBTLinkDecodingStat>>(Output{orig, "LinkErrors", 0});
  auto& decErrors = pc.outputs().make<std::vector<ChipError>>(Output{orig, "ChipErrors", 0});
  auto& errMessages = pc.outputs().make<std::vector<ErrorMessage>>(Output{orig, "ErrorInfo", 0});
  auto& physTriggers = pc.outputs().make<std::vector<PhysTrigger>>(Output{orig, "PHYSTRIG", 0});

  for (uint32_t iLayer{0}; iLayer < mLayers; ++iLayer) {
    const auto& par = AlpideParam::Instance();
    const int nROFsPerOrbit = o2::constants::lhc::LHCMaxBunches / par.getROFLengthInBC(iLayer);
    const int nROFsTF = nROFsPerOrbit * o2::base::GRPGeomHelper::getNHBFPerTF();
    int nLayer = mDoStaggering ? iLayer : -1;
    std::vector<o2::itsmft::CompClusterExt> clusCompVec;
    std::vector<o2::itsmft::ROFRecord> clusROFVec;
    std::vector<unsigned char> clusPattVec;
    std::vector<Digit> digVec;
    std::vector<GBTCalibData> calVec;
    std::vector<ROFRecord> digROFVec;
    if (mDoDigits) {
      digVec.reserve(mEstNDig[iLayer]);
      digROFVec.reserve(nROFsTF);
    }
    if (mDoClusters) {
      clusCompVec.reserve(mEstNClus[iLayer]);
      clusROFVec.reserve(nROFsTF);
      clusPattVec.reserve(mEstNClusPatt[iLayer]);
    }
    if (mDoCalibData) {
      calVec.reserve(mEstNCalib[iLayer]);
    }

    try {
      mDecoder[iLayer]->startNewTF(pc.inputs());
      mDecoder[iLayer]->setDecodeNextAuto(false);

      o2::InteractionRecord lastIR{};
      int nTriggersProcessed = mDecoder[iLayer]->getNROFsProcessed();
      static long lastErrReportTS = 0;
      while (mDecoder[iLayer]->decodeNextTrigger() >= 0) {
        if ((!lastIR.isDummy() && lastIR >= mDecoder[iLayer]->getInteractionRecord()) || mFirstIR > mDecoder[iLayer]->getInteractionRecord()) {
          const int MaxErrLog = 2;
          static int errLocCount = 0;
          if (errLocCount++ < MaxErrLog) {
            LOGP(warn, "Impossible ROF IR {}{}, previous was {}, TF 1st IR was {}, discarding in decoding", mDecoder[iLayer]->getInteractionRecord().asString(), ((mDoStaggering) ? std::format(" on layer {}", iLayer) : ""), lastIR.asString(), mFirstIR.asString());
          }
          nTriggersProcessed = 0x7fffffff; // to account for a problem with event
          continue;
        }
        lastIR = mDecoder[iLayer]->getInteractionRecord();
        mDecoder[iLayer]->fillChipsStatus(chipStatus);
        if (mDoDigits || mClusterer->getMaxROFDepthToSquash(nLayer)) { // call before clusterization, since the latter will hide the digits
          mDecoder[iLayer]->fillDecodedDigits(digVec, digROFVec);      // lot of copying involved
          if (mDoCalibData) {
            mDecoder[iLayer]->fillCalibData(calVec);
          }
        }
        if (mDoClusters && !mClusterer->getMaxROFDepthToSquash(nLayer)) { // !!! THREADS !!!
          mClusterer->process(mNThreads, *mDecoder[iLayer].get(), &clusCompVec, mDoPatterns ? &clusPattVec : nullptr, &clusROFVec);
        }
      }
      nTriggersProcessed = mDecoder[iLayer]->getNROFsProcessed() - nTriggersProcessed - 1;

      if ((nROFsTF != nTriggersProcessed) && mROFErrRepIntervalMS > 0 && mTFCounter > 1 && nTriggersProcessed > 0) {
        long currTS = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
        if (currTS - lastErrReportTS > mROFErrRepIntervalMS) {
          LOGP(critical, "Inconsistent number of ROF per TF {}{} from parameters. Received {} from readout (muting further reporting for {} ms)", nROFsTF, ((mDoStaggering) ? std::format(" on layer {}", iLayer) : ""), nTriggersProcessed, mROFErrRepIntervalMS);
          lastErrReportTS = currTS;
        }
      }
      if (mDoClusters && mClusterer->getMaxROFDepthToSquash(nLayer)) {
        // Digits squashing require to run on a batch of digits and uses a digit reader, cannot (?) run with decoder
        //  - Setup decoder for running on a batch of digits
        o2::itsmft::DigitPixelReader reader;
        reader.setSquashingDepth(mClusterer->getMaxROFDepthToSquash(nLayer));
        reader.setSquashingDist(mClusterer->getMaxRowColDiffToMask()); // Sharing same parameter/logic with masking
        reader.setMaxBCSeparationToSquash(mClusterer->getMaxBCSeparationToSquash(nLayer));
        reader.setDigits(digVec);
        reader.setROFRecords(digROFVec);
        reader.init();
        mClusterer->setMaxROFDepthToSquash(mClusterer->getMaxROFDepthToSquash(nLayer));
        mClusterer->process(mNThreads, reader, &clusCompVec, mDoPatterns ? &clusPattVec : nullptr, &clusROFVec);
      }
    } catch (const std::exception& e) {
      static size_t nErr = 0;
      auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnRawParser;
      if (++nErr < maxWarn) {
        LOGP(alarm, "EXCEPTION {} in raw decoder{}, abandoning TF decoding {}", e.what(), ((mDoStaggering) ? std::format(" on layer {}", iLayer) : ""), nErr == maxWarn ? "(will mute further warnings)" : "");
      }
    }
    if (mDoDigits) {
      pc.outputs().snapshot(Output{orig, "DIGITS", iLayer}, digVec);
      std::vector<o2::itsmft::ROFRecord> expDigRofVec(nROFsTF);
      ensureContinuousROF(digROFVec, expDigRofVec, iLayer, nROFsTF, "digits");
      pc.outputs().snapshot(Output{orig, "DIGITSROF", iLayer}, digROFVec);
      mEstNDig[iLayer] = std::max(mEstNDig[iLayer], size_t(digVec.size() * 1.2));
      if (mDoCalibData) {
        pc.outputs().snapshot(Output{orig, "GBTCALIB", iLayer}, calVec);
        mEstNCalib[iLayer] = std::max(mEstNCalib[iLayer], size_t(calVec.size() * 1.2));
      }
      LOG(debug) << mSelfName << " Decoded " << digVec.size() << " Digits in " << digROFVec.size() << " ROFs" << ((mDoStaggering) ? std::format(" on layer {}", iLayer) : "");
    }

    if (mDoClusters) { // we are not obliged to create vectors which are not requested, but other devices might not know the options of this one
      std::vector<o2::itsmft::ROFRecord> expClusRofVec(nROFsTF);
      ensureContinuousROF(clusROFVec, expClusRofVec, iLayer, nROFsTF, "clusters");
      pc.outputs().snapshot(Output{orig, "COMPCLUSTERS", iLayer}, clusCompVec);
      pc.outputs().snapshot(Output{orig, "PATTERNS", iLayer}, clusPattVec);
      pc.outputs().snapshot(Output{orig, "CLUSTERSROF", iLayer}, expClusRofVec);
      mEstNClus[iLayer] = std::max(mEstNClus[iLayer], size_t(clusCompVec.size() * 1.2));
      mEstNClusPatt[iLayer] = std::max(mEstNClusPatt[iLayer], size_t(clusPattVec.size() * 1.2));
      LOG(info) << mSelfName << " Built " << clusCompVec.size() << " clusters in " << expClusRofVec.size() << " ROFs" << ((mDoStaggering) ? std::format(" on layer {}", iLayer) : "");
    }

    mDecoder[iLayer]->collectDecodingErrors(linkErrors, decErrors, errMessages);
    physTriggers.insert(physTriggers.end(), mDecoder[iLayer]->getExternalTriggers().begin(), mDecoder[iLayer]->getExternalTriggers().end());

    if (mDumpOnError != int(GBTLink::RawDataDumps::DUMP_NONE) &&
        (!mDumpFrom1stPipeline || pc.services().get<const o2::framework::DeviceSpec>().inputTimesliceId == 0)) {
      mRawDumpedSize += mDecoder[iLayer]->produceRawDataDumps(mDumpOnError, pc.services().get<o2::framework::TimingInfo>());
      if (mRawDumpedSize > mMaxRawDumpsSize && mMaxRawDumpsSize > 0) {
        LOGP(info, "Max total dumped size {} MB exceeded allowed limit, disabling further dumping", mRawDumpedSize / (1024 * 1024));
        mDumpOnError = int(GBTLink::RawDataDumps::DUMP_NONE);
      }
    }
  }

  mTimer.Stop();
  auto tfID = pc.services().get<o2::framework::TimingInfo>().tfCounter;
  LOG(debug) << mSelfName << " Total time for TF " << tfID << '(' << mTFCounter << ") : CPU: " << mTimer.CpuTime() - timeCPU0 << " Real: " << mTimer.RealTime() - timeReal0;
  mTFCounter++;
}

///_______________________________________
template <class Mapping>
void STFDecoder<Mapping>::finalize()
{
  if (mFinalizeDone) {
    return;
  }
  mFinalizeDone = true;
  LOGF(info, "%s statistics:", mSelfName);
  LOGF(info, "%s Total STF decoding%s timing (w/o disk IO): Cpu: %.3e Real: %.3e s in %d slots", mSelfName,
       mDoClusters ? "/clustering" : "", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
  for (int iLayer{0}; iLayer < mLayers && mAllowReporting; ++iLayer) {
    if (mDecoder[iLayer]) {
      LOG_IF(info, mDoStaggering) << "Report for decoder of layer " << iLayer;
      mDecoder[iLayer]->printReport();
    }
  }
  if (mClusterer) {
    mClusterer->print();
  }
}

///_______________________________________
template <class Mapping>
void STFDecoder<Mapping>::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (pc.services().get<o2::framework::TimingInfo>().globalRunNumberChanged) { // this params need to be queried only in the beginning of the run
    pc.inputs().get<o2::itsmft::NoiseMap*>("noise");
    pc.inputs().get<o2::itsmft::DPLAlpideParam<Mapping::getDetID()>*>("alppar");
    const auto& alpParams = DPLAlpideParam<Mapping::getDetID()>::Instance();
    alpParams.printKeyValues();
    if (mDoClusters) {
      mClusterer->setContinuousReadOut(o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(Mapping::getDetID()));
      pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict");
      pc.inputs().get<o2::itsmft::ClustererParam<Mapping::getDetID()>*>("cluspar");
      // settings for the fired pixel overflow masking
      const auto& clParams = ClustererParam<Mapping::getDetID()>::Instance();
      if (clParams.maxBCDiffToMaskBias > 0 && clParams.maxBCDiffToSquashBias > 0) {
        LOGP(fatal, "maxBCDiffToMaskBias = {} and maxBCDiffToMaskBias = {} cannot be set at the same time. Either set masking or squashing with a BCDiff > 0", clParams.maxBCDiffToMaskBias, clParams.maxBCDiffToSquashBias);
      }
      clParams.printKeyValues();
      auto nbc = clParams.maxBCDiffToMaskBias;
      nbc += mClusterer->isContinuousReadOut() ? alpParams.roFrameLengthInBC : (alpParams.roFrameLengthTrig / o2::constants::lhc::LHCBunchSpacingNS);
      mClusterer->setMaxBCSeparationToMask(nbc);
      mClusterer->setMaxRowColDiffToMask(clParams.maxRowColDiffToMask);
      // Squasher
      int rofBC = mClusterer->isContinuousReadOut() ? alpParams.roFrameLengthInBC : (alpParams.roFrameLengthTrig / o2::constants::lhc::LHCBunchSpacingNS); // ROF length in BC
      mClusterer->setMaxBCSeparationToSquash(rofBC + clParams.maxBCDiffToSquashBias);
      int nROFsToSquash = 0; // squashing disabled if no reset due to maxSOTMUS>0.
      if (clParams.maxSOTMUS > 0 && rofBC > 0) {
        nROFsToSquash = 2 + int(clParams.maxSOTMUS / (rofBC * o2::constants::lhc::LHCBunchSpacingMUS)); // use squashing
      }
      mClusterer->setMaxROFDepthToSquash(clParams.maxBCDiffToSquashBias > 0 ? nROFsToSquash : 0);
      if (mDoStaggering) {
        for (int iLayer{0}; iLayer < mLayers; ++iLayer) {
          mClusterer->addMaxBCSeparationToSquash(alpParams.getROFLengthInBC(iLayer) + clParams.getMaxBCDiffToSquashBias(iLayer));
          mClusterer->addMaxROFDepthToSquash((clParams.getMaxBCDiffToSquashBias(iLayer) > 0) ? 2 + int(clParams.maxSOTMUS / (alpParams.getROFLengthInBC(iLayer) * o2::constants::lhc::LHCBunchSpacingMUS)) : 0);
        }
      }
      mClusterer->print(false);
    }
  }
  mFirstTFOrbit = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
  mFirstIR = o2::InteractionRecord(0, mFirstTFOrbit);
}

///_______________________________________
template <class Mapping>
void STFDecoder<Mapping>::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher(Mapping::getOrigin(), "NOISEMAP", 0)) {
    LOG(info) << Mapping::getName() << " noise map updated" << (!mApplyNoiseMap ? " but masking is disabled" : "");
    if (mApplyNoiseMap) {
      AlpideCoder::setNoisyPixels((const NoiseMap*)obj);
    }
    return;
  }
  if (matcher == ConcreteDataMatcher(Mapping::getOrigin(), "CLUSDICT", 0)) {
    LOG(info) << Mapping::getName() << " cluster dictionary updated" << (!mUseClusterDictionary ? " but its using is disabled" : "");
    if (mUseClusterDictionary) {
      mClusterer->setDictionary((const TopologyDictionary*)obj);
    }
    return;
  }
  // Note: strictly speaking, for Configurable params we don't need finaliseCCDB check, the singletons are updated at the CCDB fetcher level
  if (matcher == ConcreteDataMatcher(Mapping::getOrigin(), "ALPIDEPARAM", 0)) {
    LOG(info) << "Alpide param updated";
    return;
  }
}

///_______________________________________
template <class Mapping>
void STFDecoder<Mapping>::reset()
{
  // reset for the new run
  mFinalizeDone = false;
  mTFCounter = 0;
  mTimer.Reset();
  for (int iLayer{0}; iLayer < mLayers; ++iLayer) {
    if (mDecoder[iLayer]) {
      mDecoder[iLayer]->reset();
    }
  }
  if (mClusterer) {
    mClusterer->reset();
  }
}

///_______________________________________
template <class Mapping>
void STFDecoder<Mapping>::ensureContinuousROF(const std::vector<ROFRecord>& rofVec, std::vector<ROFRecord>& expROFVec, int lr, int nROFsTF, const char* name)
{
  const auto& par = AlpideParam::Instance();
  // ensure that the rof output is continuous
  // we will preserve the digits/clusters as they are but the stray ROFs will be removed (leaving their clusters/digits unaddressed).
  expROFVec.clear();
  expROFVec.resize(nROFsTF);
  for (int iROF{0}; iROF < nROFsTF; ++iROF) {
    auto& rof = expROFVec[iROF];
    int orb = iROF * par.getROFLengthInBC(lr) / o2::constants::lhc::LHCMaxBunches + mFirstTFOrbit;
    int bc = iROF * par.getROFLengthInBC(lr) % o2::constants::lhc::LHCMaxBunches + par.getROFDelayInBC(lr);
    o2::InteractionRecord ir(bc, orb);
    rof.setBCData(ir);
    rof.setROFrame(iROF);
    rof.setNEntries(0);
    rof.setFirstEntry(-1);
  }
  uint32_t prevEntry{0};
  for (const auto& rof : rofVec) {
    const auto& ir = rof.getBCData();
    if (ir < mFirstIR) {
      LOGP(warn, "Discard ROF {} preceding TF 1st orbit {}{}", ir.asString(), mFirstTFOrbit, ((mDoStaggering) ? std::format(" on layer {}", lr) : ""));
      continue;
    }
    auto irToFirst = ir - mFirstIR;
    if (irToFirst.toLong() - par.getROFDelayInBC(lr) < 0) {
      LOGP(warn, "Discard ROF {} preceding TF 1st orbit {} due to imposed ROF delay{}", ir.asString(), mFirstTFOrbit, ((mDoStaggering) ? std::format(" on layer {}", lr) : ""));
      continue;
    }
    irToFirst -= par.getROFDelayInBC(lr);
    const long irROF = irToFirst.toLong() / par.getROFLengthInBC(lr);
    if (irROF >= nROFsTF) {
      LOGP(warn, "Discard ROF {} exceeding TF orbit range{}", ir.asString(), ((mDoStaggering) ? std::format(" on layer {}", lr) : ""));
      continue;
    }
    auto& expROF = expROFVec[irROF];
    if (expROF.getNEntries() == 0) {
      expROF.setFirstEntry(rof.getFirstEntry());
      expROF.setNEntries(rof.getNEntries());
    } else {
      if (expROF.getNEntries() < rof.getNEntries()) {
        LOGP(warn, "Repeating {} with {} {}, prefer to already processed instance with {} {}{}", rof.asString(), rof.getNEntries(), name, expROF.getNEntries(), name, ((mDoStaggering) ? std::format(" on layer {}", lr) : ""));
        expROF.setFirstEntry(rof.getFirstEntry());
        expROF.setNEntries(rof.getNEntries());
      } else {
        LOGP(warn, "Repeating {} with {} {}, discard preferring already processed instance with {} {}{}", rof.asString(), rof.getNEntries(), name, expROF.getNEntries(), name, ((mDoStaggering) ? std::format(" on layer {}", lr) : ""));
      }
    }
  }
  int prevFirst{0};
  for (auto& rof : expROFVec) {
    if (rof.getFirstEntry() < 0) {
      rof.setFirstEntry(prevFirst);
    }
    prevFirst = rof.getFirstEntry();
  }
}

///_______________________________________
DataProcessorSpec getSTFDecoderSpec(const STFDecoderInp& inp)
{
  std::vector<OutputSpec> outputs;
  auto inputs = o2::framework::select(inp.inputSpec.c_str());
  uint32_t nLayers = 1;
  if (inp.origin == o2::header::gDataOriginITS && inp.doStaggering) {
    nLayers = DPLAlpideParam<o2::detectors::DetID::ITS>::getNLayers();
  } else if (inp.origin == o2::header::gDataOriginMFT && inp.doStaggering) {
    nLayers = DPLAlpideParam<o2::detectors::DetID::MFT>::getNLayers();
  }
  for (uint32_t iLayer = 0; iLayer < nLayers; ++iLayer) {
    if (inp.doDigits) {
      outputs.emplace_back(inp.origin, "DIGITS", iLayer, Lifetime::Timeframe);
      outputs.emplace_back(inp.origin, "DIGITSROF", iLayer, Lifetime::Timeframe);
    }
    if (inp.doClusters) {
      outputs.emplace_back(inp.origin, "COMPCLUSTERS", iLayer, Lifetime::Timeframe);
      outputs.emplace_back(inp.origin, "CLUSTERSROF", iLayer, Lifetime::Timeframe);
      // in principle, we don't need to open this input if we don't need to send real data,
      // but other devices expecting it do not know about options of this device: problem?
      // if (doClusters && doPatterns)
      outputs.emplace_back(inp.origin, "PATTERNS", iLayer, Lifetime::Timeframe);
    }
  }
  if (inp.doDigits && inp.doCalib) {
    outputs.emplace_back(inp.origin, "GBTCALIB", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back(inp.origin, "PHYSTRIG", 0, Lifetime::Timeframe);
  outputs.emplace_back(inp.origin, "LinkErrors", 0, Lifetime::Timeframe);
  outputs.emplace_back(inp.origin, "ChipErrors", 0, Lifetime::Timeframe);
  outputs.emplace_back(inp.origin, "ErrorInfo", 0, Lifetime::Timeframe);
  outputs.emplace_back(inp.origin, "CHIPSSTATUS", 0, Lifetime::Timeframe);

  if (inp.askSTFDist) {
    // request the input FLP/DISTSUBTIMEFRAME/0 that is _guaranteed_ to be present, even if none of our raw data is present.
    inputs.emplace_back("stfDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }
  inputs.emplace_back("noise", inp.origin, "NOISEMAP", 0, Lifetime::Condition,
                      o2::framework::ccdbParamSpec(fmt::format("{}/Calib/NoiseMap", inp.origin.as<std::string>())));
  inputs.emplace_back("alppar", inp.origin, "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Config/AlpideParam", inp.origin.as<std::string>())));
  if (inp.doClusters) {
    inputs.emplace_back("cldict", inp.origin, "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Calib/ClusterDictionary", inp.origin.as<std::string>())));
    inputs.emplace_back("cluspar", inp.origin, "CLUSPARAM", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Config/ClustererParam", inp.origin.as<std::string>())));
  }

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                              true,                           // GRPECS=true
                                                              false,                          // GRPLHCIF
                                                              false,                          // GRPMagField
                                                              false,                          // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry
                                                              inputs,
                                                              true); // query only once all objects except mag.field

  return DataProcessorSpec{
    .name = inp.deviceName,
    .inputs = inputs,
    .outputs = outputs,
    .algorithm = inp.origin == o2::header::gDataOriginITS ? AlgorithmSpec{adaptFromTask<STFDecoder<ChipMappingITS>>(inp, ggRequest)} : AlgorithmSpec{adaptFromTask<STFDecoder<ChipMappingMFT>>(inp, ggRequest)},
    .options = Options{
      {"nthreads", VariantType::Int, 1, {"Number of decoding/clustering threads"}},
      {"decoder-verbosity", VariantType::Int, 0, {"Verbosity level (-1: silent, 0: errors, 1: headers, 2: data, 3: raw data dump) of 1st lane"}},
      {"always-parse-trigger", VariantType::Bool, false, {"parse trigger word even if flags continuation of old trigger"}},
      {"raw-data-dumps", VariantType::Int, int(GBTLink::RawDataDumps::DUMP_NONE), {"Raw data dumps on error (0: none, 1: HBF for link, 2: whole TF for all links. If negative, dump only on from 1st pipeline."}},
      {"raw-data-dumps-directory", VariantType::String, "", {"Destination directory for the raw data dumps"}},
      {"stop-raw-data-dumps-after-size", VariantType::Int, 1024, {"Stop dumping once this size in MB is accumulated. 0: no limit"}},
      {"unmute-extra-lanes", VariantType::Bool, false, {"allow extra lanes to be as verbose as 1st one"}},
      {"allow-empty-rofs", VariantType::Bool, false, {"record ROFs w/o any hit"}},
      {"ignore-noise-map", VariantType::Bool, false, {"do not mask pixels flagged in the noise map"}},
      {"accept-rof-rampup-data", VariantType::Bool, false, {"do not discard data during ROF ramp up"}},
      {"rof-length-error-freq", VariantType::Float, 60.f, {"do not report ROF length error more frequently than this value, disable if negative"}},
      {"ignore-cluster-dictionary", VariantType::Bool, false, {"do not use cluster dictionary, always store explicit patterns"}}}};
}

} // namespace itsmft
} // namespace o2
