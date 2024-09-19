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

// @brief Aux.class initialize HBFUtils
// @author ruben.shahoyan@cern.ch

#include "Headers/DataHeader.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DetectorsRaw/HBFUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/StringUtils.h"
#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigParamsHelper.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CallbackService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessingHeader.h"
#include <gsl/span>
#include <TFile.h>
#include <TGrid.h>

using namespace o2::raw;
namespace o2f = o2::framework;

/// If the workflow has devices w/o inputs, we assume that these are data readers in root-file based workflow.
/// In this case this class will configure these devices DataHeader.firstTForbit generator to provide orbit according to HBFUtil setings
/// In case the configcontext has relevant option, the HBFUtils will be beforehand updated from the file indicated by this option.
/// (only those fields of HBFUtils which were not modified before, e.g. by ConfigurableParam::updateFromString)

int HBFUtilsInitializer::NTFs = 0;
long HBFUtilsInitializer::LastIRFrameIndex = -1;
bool HBFUtilsInitializer::LastIRFrameSplit = false;
std::vector<o2::dataformats::IRFrame> HBFUtilsInitializer::IRFrames = {};
o2::dataformats::IRFrame HBFUtilsInitializer::IRFrameSel = {};

//_________________________________________________________
HBFUtilsInitializer::HBFUtilsInitializer(const o2f::ConfigContext& configcontext, o2f::WorkflowSpec& wf)
{
  bool upstream = false; // timing info will be provided from upstream readers-driver, just subscribe to it
  std::string rootFileInput{};
  std::string hbfuInput{};

  auto updateHBFUtils = [&configcontext, &upstream, &hbfuInput, &rootFileInput]() {
    static bool done = false;
    if (!done) {
      bool helpasked = configcontext.helpOnCommandLine(); // if help is asked, don't take for granted that the ini file is there, don't produce an error if it is not!
      auto conf = configcontext.options().isSet(HBFConfOpt) ? configcontext.options().get<std::string>(HBFConfOpt) : "";
      if (!conf.empty()) {
        auto vopts = o2::utils::Str::tokenize(conf, ',');
        for (const auto& optStr : vopts) {
          if (optStr == UpstreamOpt) {
            upstream = true;
            continue;
          }
          HBFOpt opt = getOptType(optStr);
          if ((opt == HBFOpt::INI || opt == HBFOpt::JSON) && !helpasked) {
            o2::conf::ConfigurableParam::updateFromFile(optStr, "HBFUtils", true); // update only those values which were not touched yet (provenance == kCODE)
            const auto& hbfu = o2::raw::HBFUtils::Instance();
            hbfu.checkConsistency();
            hbfuInput = optStr;
          } else if (opt == HBFOpt::HBFUTILS) {
            const auto& hbfu = o2::raw::HBFUtils::Instance();
            hbfu.checkConsistency();
            hbfuInput = optStr;
          } else if (opt == HBFOpt::ROOT) {
            rootFileInput = optStr;
          }
        }
      }
      done = true;
      /* RSS
      if (upstream) {
  if (rootFileInput.empty()) {
    throw std::runtime_error(fmt::format("invalid option {}: upstream can be used only with root file providing TFIDInfo or IRFrames", conf));
  }
      }
      */
    }
  };

  if (configcontext.options().hasOption("disable-root-input") && configcontext.options().get<bool>("disable-root-input")) {
    return; // we apply HBFUtilsInitializer only in case of root readers
  }
  const auto& hbfu = o2::raw::HBFUtils::Instance();
  for (auto& spec : wf) {
    if (spec.inputs.empty()) {
      updateHBFUtils();
      if (!upstream || spec.name == ReaderDriverDevice) {
        if (!hbfuInput.empty()) { // MC timing coming from the HBFUtils configurable
          o2f::ConfigParamsHelper::addOptionIfMissing(spec.options, o2f::ConfigParamSpec{HBFTFInfoOpt, o2f::VariantType::String, HBFUSrc, {"HBFUtils input"}});
          if (!rootFileInput.empty() && spec.name == ReaderDriverDevice) { // this is IRFrame file passed to reader driver
            o2f::ConfigParamsHelper::addOptionIfMissing(spec.options, o2f::ConfigParamSpec{HBFIRFrameOpt, o2f::VariantType::String, rootFileInput, {"root file with selected IR-frames"}});
          }
        } else {
          o2f::ConfigParamsHelper::addOptionIfMissing(spec.options, o2f::ConfigParamSpec{HBFTFInfoOpt, o2f::VariantType::String, rootFileInput, {"root file with per-TF info"}});
        }
        o2f::ConfigParamsHelper::addOptionIfMissing(spec.options, o2f::ConfigParamSpec{DelayOpt, o2f::VariantType::Float, 0.f, {"delay in seconds between consecutive TFs sending"}});
      } else { // subsribe to upstream timing info from readers-driver
        spec.inputs.emplace_back(o2f::InputSpec{"driverInfo", "GLO", "READER_DRIVER", 0, o2f::Lifetime::Timeframe});
        if (!hbfuInput.empty() && !rootFileInput.empty()) { // flag that the READER_DRIVER is not dummy but contains IRFrames
          o2f::ConfigParamsHelper::addOptionIfMissing(spec.options, o2f::ConfigParamSpec{IgnoreIRFramesOpt, o2f::VariantType::Bool, false, {"ignore IR-frames info"}});
        }
      }
    }
  }
}

//_________________________________________________________
HBFUtilsInitializer::HBFOpt HBFUtilsInitializer::getOptType(const std::string& optString)
{
  // return type of the file provided via HBFConfOpt
  HBFOpt opt = HBFOpt::NONE;
  if (!optString.empty()) {
    if (o2::utils::Str::endsWith(optString, ".ini")) {
      opt = HBFOpt::INI;
    } else if (o2::utils::Str::endsWith(optString, ".json")) {
      opt = HBFOpt::JSON;
    } else if (o2::utils::Str::endsWith(optString, ".root")) {
      opt = HBFOpt::ROOT;
    } else if (optString == HBFUSrc) {
      opt = HBFOpt::HBFUTILS;
    } else if (optString != "none") {
      throw std::runtime_error(fmt::format("invalid option {} for {}", optString, HBFConfOpt));
    }
  }
  return opt;
}

//_________________________________________________________
std::vector<o2::dataformats::TFIDInfo> HBFUtilsInitializer::readTFIDInfoVector(const std::string& fname)
{
  if (o2::utils::Str::beginsWith(fname, "alien://") && !gGrid && !TGrid::Connect("alien://")) {
    LOGP(fatal, "could not open alien connection to read {}", fname);
  }
  std::unique_ptr<TFile> fl(TFile::Open(fname.c_str()));
  auto vptr = (std::vector<o2::dataformats::TFIDInfo>*)fl->GetObjectChecked("tfidinfo", "std::vector<o2::dataformats::TFIDInfo>");
  if (!vptr) {
    throw std::runtime_error(fmt::format("Failed to read tfidinfo vector from {}", fname));
  }
  std::vector<o2::dataformats::TFIDInfo> v(*vptr);
  NTFs = v.size();
  return v;
}

//_________________________________________________________
void HBFUtilsInitializer::readIRFramesVector(const std::string& fname)
{
  if (o2::utils::Str::beginsWith(fname, "alien://") && !gGrid && !TGrid::Connect("alien://")) {
    LOGP(fatal, "could not open alien connection to read {}", fname);
  }
  std::unique_ptr<TFile> fl(TFile::Open(fname.c_str()));
  auto vptr = (std::vector<o2::dataformats::IRFrame>*)fl->GetObjectChecked("irframes", "std::vector<o2::dataformats::IRFrame>");
  if (!vptr) {
    throw std::runtime_error(fmt::format("Failed to read irframes vector from {}", fname));
  }
  std::vector<o2::dataformats::IRFrame> v(*vptr);
  NTFs = vptr->size();
  LastIRFrameIndex = -1;
  IRFrames.swap(*vptr);
}

//_________________________________________________________
void HBFUtilsInitializer::assignDataHeaderFromTFIDInfo(const std::vector<o2::dataformats::TFIDInfo>& tfinfoVec, o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph)
{
  const auto tfinf = tfinfoVec[dh.tfCounter % tfinfoVec.size()];
  LOGP(debug, "Setting DH for {}/{} from tfCounter={} firstTForbit={} runNumber={} to tfCounter={} firstTForbit={} runNumber={}",
       dh.dataOrigin.as<std::string>(), dh.dataDescription.as<std::string>(), dh.tfCounter, dh.firstTForbit, dh.runNumber, tfinf.tfCounter, tfinf.firstTForbit, tfinf.runNumber);
  dh.firstTForbit = tfinf.firstTForbit;
  dh.tfCounter = tfinf.tfCounter;
  dh.runNumber = tfinf.runNumber;
  dph.creation = tfinf.creation;
}

//_________________________________________________________
void HBFUtilsInitializer::assignDataHeaderFromHBFUtils(o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph)
{
  const auto& hbfu = o2::raw::HBFUtils::Instance();
  auto offset = hbfu.getFirstIRofTF({0, hbfu.orbitFirstSampled}).orbit;
  dh.firstTForbit = offset + hbfu.nHBFPerTF * dh.tfCounter;
  dh.runNumber = hbfu.runNumber;
  dph.creation = hbfu.startTime + (dh.firstTForbit - hbfu.orbitFirst) * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
  LOGP(debug, "SETTING DH for {}/{} from tfCounter={} firstTForbit={} runNumber={}",
       dh.dataOrigin.as<std::string>(), dh.dataDescription.as<std::string>(), dh.tfCounter, dh.firstTForbit, dh.runNumber);
}

//_________________________________________________________
void HBFUtilsInitializer::assignDataHeaderFromHBFUtilWithIRFrames(o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph)
{
  const auto& hbfu = o2::raw::HBFUtils::Instance();
  static int64_t offset = hbfu.getFirstIRofTF({0, hbfu.orbitFirstSampled}).orbit;
  dh.runNumber = hbfu.runNumber;
  dh.firstTForbit = offset + hbfu.nHBFPerTF * dh.tfCounter; // fallback settings
  IRFrameSel.getMin().clear();                              // invalidate
  if (LastIRFrameSplit) {                                   // previously sent IRFrame has a continuation in the next TF
    LastIRFrameIndex--;
    LastIRFrameSplit = false;
  }
  o2::InteractionRecord ir0Mn, ir0Mx;
  while (++LastIRFrameIndex < NTFs) {
    IRFrameSel = IRFrames[LastIRFrameIndex]; // copy!
    ir0Mn = hbfu.getFirstIRofTF(IRFrameSel.getMin());
    ir0Mx = hbfu.getFirstIRofTF(IRFrameSel.getMax());
    if (ir0Mn.orbit < offset) {
      if (ir0Mx.orbit < offset) {
        LOGP(warn, "IRFrame#{} ({}-{}) precedes 1st sampled TF {}, skipping",
             LastIRFrameIndex,
             IRFrameSel.getMin().asString(),
             IRFrameSel.getMax().asString(),
             hbfu.getFirstIRofTF({0, hbfu.orbitFirstSampled}).asString());
        IRFrameSel.getMin().clear();
        continue;
      } else {
        LOGP(warn, "IRFrame#{} ({}-{}) start precedes 1st sampled TF {}, overriding start", LastIRFrameIndex,
             IRFrameSel.getMin().asString(),
             IRFrameSel.getMax().asString(),
             hbfu.getFirstIRofTF({0, hbfu.orbitFirstSampled}).asString());
        ir0Mn = hbfu.getFirstIRofTF({0, hbfu.orbitFirstSampled});
        IRFrameSel.setMin(ir0Mn);
      }
    }
    if (IRFrameSel.getMax().orbit >= ir0Mn.orbit + hbfu.nHBFPerTF) {
      ir0Mx = ir0Mn + int64_t(hbfu.nHBFPerTF) * o2::constants::lhc::LHCMaxBunches - 1;
      LOGP(warn, "IRFrame#{} ({}-{}) is split to ({}-{}) and ({}-{})",
           LastIRFrameIndex,
           IRFrameSel.getMin().asString(), IRFrameSel.getMax().asString(),
           IRFrameSel.getMin().asString(), ir0Mx.asString(),
           (ir0Mx + 1).asString(), IRFrameSel.getMax().asString());
      IRFrameSel.setMax(ir0Mx);
      IRFrames[LastIRFrameIndex].setMin(ir0Mx + 1);
      LastIRFrameSplit = true;
    }
    break;
  }
  if (IRFrameSel.getMin().isDummy() || IRFrameSel.getMax().isDummy()) {
    LOGP(warn, "Failed to define IRFrame");
  } else {
    dh.tfCounter = (ir0Mn.orbit - offset) / hbfu.nHBFPerTF;
    dh.firstTForbit = ir0Mn.orbit;
    if (LastIRFrameIndex == NTFs - 1 && !LastIRFrameSplit) {
      IRFrameSel.setLast();
    }
  }
  dph.creation = hbfu.startTime + (dh.firstTForbit - hbfu.orbitFirst) * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
  LOGP(debug, "SETTING DH for {}/{} from tfCounter={} firstTForbit={} runNumber={} and IRFrame ({}-{})",
       dh.dataOrigin.as<std::string>(), dh.dataDescription.as<std::string>(), dh.tfCounter, dh.firstTForbit, dh.runNumber,
       IRFrameSel.getMin().asString(), IRFrameSel.getMax().asString());
}

//_________________________________________________________
void HBFUtilsInitializer::addNewTimeSliceCallback(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  policies.push_back(o2::framework::CallbacksPolicy{
    [](o2::framework::DeviceSpec const& spec, o2::framework::ConfigContext const& context) -> bool {
      return (!context.helpOnCommandLine() && o2f::ConfigParamsHelper::hasOption(spec.options, HBFTFInfoOpt));
    },
    [](o2::framework::CallbackService& service, o2::framework::InitContext& context) {
      std::string irFrames, tfInput = context.options().get<std::string>(HBFTFInfoOpt);
      if (context.options().hasOption(HBFIRFrameOpt)) {
        irFrames = context.options().get<std::string>(HBFIRFrameOpt);
      }
      uint32_t delay = context.options().hasOption(DelayOpt) && context.options().isSet(DelayOpt) ? uint32_t(1e6 * context.options().get<float>(DelayOpt)) : 0;
      if (!tfInput.empty()) {
        if (tfInput == HBFUSrc) { // simple linear enumeration from already updated HBFUtils
          if (irFrames.empty()) { // push the whole TF
            NTFs = 1;
            service.set<o2::framework::CallbackService::Id::NewTimeslice>([delay](o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph) {
              assignDataHeaderFromHBFUtils(dh, dph);
              static size_t tfcount = 0;
              if (tfcount++ && delay > 0) {
                usleep(delay);
              }
            });
          } else { // linear enumeration with IRFrames selection (skimming)
            if (!o2::utils::Str::pathExists(irFrames)) {
              throw std::runtime_error(fmt::format("file {} does not exist", irFrames));
            }
            readIRFramesVector(irFrames);
            service.set<o2::framework::CallbackService::Id::NewTimeslice>(
              [delay](o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph) {
                assignDataHeaderFromHBFUtilWithIRFrames(dh, dph);
                static size_t tfcount = 0;
                if (tfcount++ && delay > 0) {
                  usleep(delay);
                }
              });
          }
        } else if (o2::utils::Str::endsWith(tfInput, ".root")) { // read TFIDinfo from file
          if (!o2::utils::Str::pathExists(tfInput)) {
            throw std::runtime_error(fmt::format("file {} does not exist", tfInput));
          }
          service.set<o2::framework::CallbackService::Id::NewTimeslice>(
            [tfidinfo = readTFIDInfoVector(tfInput), delay](o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph) {
              assignDataHeaderFromTFIDInfo(tfidinfo, dh, dph);
              static size_t tfcount = 0;
              if (tfcount++ && delay > 0) {
                usleep(delay);
              }
            });
        } else { // do not modify timing info
          // we may remove the highest bit set on the creation time?
        }
      }
    }});
}

void HBFUtilsInitializer::addConfigOption(std::vector<o2f::ConfigParamSpec>& opts, const std::string& defOpt)
{
  o2f::ConfigParamsHelper::addOptionIfMissing(opts, o2f::ConfigParamSpec{HBFConfOpt, o2f::VariantType::String, defOpt, {R"(ConfigurableParam ini file or "hbfutils" for HBFUtils, root file with per-TF info (augmented with ,upstream if reader-driver is used) or "none")"}});
}
