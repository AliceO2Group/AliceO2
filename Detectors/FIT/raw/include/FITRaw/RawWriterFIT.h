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
//
// file RawWriterFIT.h Base class for RAW data writing
//
// Artur.Furs
// afurs@cern.ch

#ifndef ALICEO2_FIT_RAWREADERBASEFIT_H_
#define ALICEO2_FIT_RAWREADERBASEFIT_H_
#include <iostream>
#include <vector>
#include <cstdlib>
#include <Rtypes.h>
#include <TStopwatch.h>
#include <boost/program_options.hpp>
#include <gsl/span>
#include <fmt/format.h>

#include "FITRaw/DataBlockFIT.h"
#include "FITRaw/DigitBlockFIT.h"
#include <DataFormatsFIT/RawDataMetric.h>
#include <Framework/Logger.h>
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"

namespace o2
{
namespace fit
{
// Raw writer for FIT, based on RawReader
template <typename DigitBlockFITtype, typename DataBlockPMtype, typename DataBlockTCMtype, typename = typename std::enable_if_t<DataBlockPMtype::sIsPadded == DataBlockTCMtype::sIsPadded>>
class RawWriterFIT
{
 public:
  typedef DigitBlockFITtype DigitBlockFIT_t;
  typedef typename DigitBlockFIT_t::LookupTable_t LookupTable_t;
  typedef typename LookupTable_t::Topo_t Topo_t;
  typedef DataBlockPMtype DataBlockPM_t;
  typedef DataBlockTCMtype DataBlockTCM_t;
  typedef typename RawDataMetric::Status_t MetricStatus_t;
  using RDHUtils = o2::raw::RDHUtils;
  RawWriterFIT() = default;
  ~RawWriterFIT() = default;
  o2::raw::RawFileWriter& getWriter() { return mWriter; }
  void setFileFor(const std::string& fileFor) { mFileFor = fileFor; }
  void setFlpName(const std::string& flpName) { mFlpName = flpName; }
  bool getFilePerLink() const { return mOutputPerLink; }
  void setStatusEmu(MetricStatus_t statusEmu) { mStatusEmu = statusEmu; }
  void setRandomEmu(bool isRandomStatusEmu) { mIsRandomStatusEmu = isRandomStatusEmu; }

  void setVerbosity(int verbosityLevel) { mVerbosity = verbosityLevel; }
  void setCCDBurl(const std::string& ccdbPath) { LookupTable_t::setCCDBurl(ccdbPath); }
  void setLUTpath(const std::string& lutPath) { LookupTable_t::setLUTpath(lutPath); }
  int getVerbosity() const { return mVerbosity; }
  int carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                      const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const
  {
    return 0; // do not split, always start new CRU page
  }
  void convertDigitsToRaw(const std::string& outputDir, const std::string& filenameDigits, long timestamp = -1)
  {
    LOG(info) << "Converting Digits to Raw data...";
    if (mStatusEmu > 0 || mIsRandomStatusEmu) {
      LOG(warning) << "Raw data metric emulation applied: isRandom " << mIsRandomStatusEmu << "; metricStatus: " << static_cast<uint16_t>(mStatusEmu);
      if (mStatusEmu == 0) {
        LOG(warning) << "Metric status is not specified, all bits will be randomly generated";
        mStatusEmu = RawDataMetric::getAllBitsActivated();
      }
    }
    mWriter.setCarryOverCallBack(this);
    LookupTable_t::Instance(nullptr, timestamp).printFullMap();
    // Preparing topo2FEEmetadata map
    mMapTopo2FEEmetadata.clear();
    mMapTopo2FEEmetadata = LookupTable_t::Instance().template makeMapFEEmetadata<o2::header::RAWDataHeader, RDHUtils>();
    // Preparing filenames
    std::string detName = LookupTable_t::sDetectorName;
    auto makeFilename = [&](const o2::header::RAWDataHeader& rdh) -> std::string {
      std::string maskName{};
      if (mFileFor != "all") { // single file for all links
        maskName += fmt::format("_{}", mFlpName);
        if (mFileFor != "flp") {
          maskName += fmt::format("_cru{}_{}", RDHUtils::getCRUID(rdh), RDHUtils::getEndPointID(rdh));
          if (mFileFor != "cruendpoint") {
            maskName += fmt::format("_lnk{}_feeid{}", RDHUtils::getLinkID(rdh), RDHUtils::getFEEID(rdh));
            if (mFileFor != "link") {
              throw std::runtime_error("invalid option provided for file grouping");
            }
          }
        }
      }
      std::string outputFilename = o2::utils::Str::concat_string(outputDir, detName, maskName, ".raw");
      return outputFilename;
    };
    // Registering links
    for (const auto& metadataPair : mMapTopo2FEEmetadata) {
      const auto& rdh = metadataPair.second;
      const auto outputFilename = makeFilename(rdh);
      mWriter.registerLink(RDHUtils::getFEEID(rdh), RDHUtils::getCRUID(rdh), RDHUtils::getLinkID(rdh), RDHUtils::getEndPointID(rdh), outputFilename);
    }
    // Processing digits into raw data
    TFile* inputFile = TFile::Open(filenameDigits.c_str());
    assert(inputFile != nullptr);
    LOG(info) << "Source file: " << filenameDigits;
    TTree* inputTree = dynamic_cast<TTree*>(inputFile->Get("o2sim"));
    DigitBlockFIT_t::processDigitBlocks(inputTree, *this);
    delete inputTree;
    inputFile->Close();
    delete inputFile;
  }
  void processDigitBlockPerTF(const std::vector<DigitBlockFIT_t>& vecDigitBlock) // Is used in DigitBlockFIT_t::processDigitBlocks for each TF (TTree entry)
  {
    for (const auto& digitBlock : vecDigitBlock) {
      MetricStatus_t statusEmu{mStatusEmu};
      // Processing PM data
      if (mIsRandomStatusEmu) {
        statusEmu = static_cast<MetricStatus_t>(std::rand()) & mStatusEmu;
      }
      const auto mapDataBlockPM = digitBlock.template decomposeDigits<DataBlockPM_t>(statusEmu);
      if (mVerbosity > 0) {
        digitBlock.print();
      }
      for (const auto& dataBlockPair : mapDataBlockPM) {
        const auto& topo = dataBlockPair.first;
        const auto& dataBlock = dataBlockPair.second;
        const auto& itRdh = mMapTopo2FEEmetadata.find(topo);
        if (itRdh == mMapTopo2FEEmetadata.end()) {
          LOG(warning) << "No CRU entry in map! Data block: ";
          dataBlock.print();
          continue;
        }
        if (mVerbosity > 0) {
          dataBlock.print();
        }
        const auto& rdh = itRdh->second;
        auto data = dataBlock.serialize();
        mWriter.addData(RDHUtils::getFEEID(rdh), RDHUtils::getCRUID(rdh), RDHUtils::getLinkID(rdh), RDHUtils::getEndPointID(rdh), dataBlock.getInteractionRecord(), data);
      }
      // Processing TCM data
      if (mIsRandomStatusEmu) {
        statusEmu = static_cast<MetricStatus_t>(std::rand()) & mStatusEmu;
      }
      const auto dataBlockPair = digitBlock.template decomposeDigits<DataBlockTCM_t>(statusEmu);
      const auto& topo = dataBlockPair.first;
      const auto& dataBlock = dataBlockPair.second;
      const auto& itRdh = mMapTopo2FEEmetadata.find(topo);
      if (itRdh == mMapTopo2FEEmetadata.end()) {
        LOG(warning) << "No CRU entry in map! Data block: ";
        dataBlock.print();
        continue;
      }
      if (mVerbosity > 0) {
        dataBlock.print();
      }
      const auto& rdh = itRdh->second;
      auto data = dataBlock.serialize();
      mWriter.addData(RDHUtils::getFEEID(rdh), RDHUtils::getCRUID(rdh), RDHUtils::getLinkID(rdh), RDHUtils::getEndPointID(rdh), dataBlock.getInteractionRecord(), data);
    }
  }
  static constexpr auto getDetID() { return LookupTable_t::sDetID; };
  o2::raw::RawFileWriter mWriter{getDetID().getDataOrigin()};
  std::string mFlpName{};
  std::string mFileFor{};
  MetricStatus_t mStatusEmu{};    // for metrics emulation, contains bits RawDataMetric::EStatusBits
  bool mIsRandomStatusEmu{false}; // to use status bits randomly
  int mDataFormat{0};             // RDH::dataFormat field, 0 - padded, 2 - no padding
  std::map<Topo_t, o2::header::RAWDataHeader> mMapTopo2FEEmetadata;
  // const o2::raw::HBFUtils& mSampler = o2::raw::HBFUtils::Instance();
  bool mOutputPerLink = false;
  int mVerbosity = 0;
};

/*
 * Special configurator for digit2raw executables
 */
namespace bpo = boost::program_options;
struct DigitToRawConfig {
  using VariablesMap = typename boost::program_options::variables_map;
  using OptionsDescription = typename boost::program_options::options_description;
  using MetricStatus_t = typename RawDataMetric::Status_t;
  DigitToRawConfig(const VariablesMap& vm)
  {
    setFromExecOptions(vm);
  }
  DigitToRawConfig() = default; // careful, data format should be then checked via adjustDataFormat()
  void setFromExecOptions(const VariablesMap& vm)
  {
    mVerbosity = vm["verbosity"].as<int>();
    mInputFile = vm["input-file"].as<std::string>();
    mOutputDir = vm["output-dir"].as<std::string>();
    mFileFor = vm["file-for"].as<std::string>();
    mFlpName = vm["flp-name"].as<std::string>();
    mConfigKeyValues = vm["configKeyValues"].as<std::string>();

    mRdhVersion = vm["rdh-version"].as<uint32_t>();
    mNoEmptyHBF = vm["no-empty-hbf"].as<bool>();
    mConfDig = vm["hbfutils-config"].as<std::string>();
    mEnablePadding = vm["enable-padding"].as<bool>();
    mCcdbPath = vm["ccdb-path"].as<std::string>();
    mChannelMapPath = vm["lut-path"].as<std::string>();

    mMetricStatusEmu = static_cast<MetricStatus_t>(vm["emu-rawdata-metrics"].as<uint64_t>());
    mIsRandomMetric = vm["emu-rawdata-metrics-random"].as<bool>();

    adjustDataFormat();
  }
  static void configureExecOptions(OptionsDescription& opt_general, const std::string& defaultInputFilename, const std::string& defaultFLP_name)
  {
    auto add_option = opt_general.add_options();
    // common configs
    add_option("verbosity,v", bpo::value<int>()->default_value(0), "verbosity level");
    add_option("input-file,i", bpo::value<std::string>()->default_value(defaultInputFilename), "input digits file");
    add_option("output-dir,o", bpo::value<std::string>()->default_value("./"), "output directory for raw data");
    add_option("flp-name", bpo::value<std::string>()->default_value(defaultFLP_name), "single file per: all,flp,cru,link");
    add_option("file-for,f", bpo::value<std::string>()->default_value("all"), "single file per: all,flp,cruendpoint,link");
    add_option("configKeyValues", bpo::value<std::string>()->default_value(""), "comma-separated configKeyValues");
    // raw data configs
    uint32_t defRDH = o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>();
    add_option("rdh-version,r", bpo::value<uint32_t>()->default_value(defRDH), "RDH version to use");
    add_option("no-empty-hbf,e", bpo::value<bool>()->default_value(false)->implicit_value(true), "do not create empty HBF pages (except for HBF starting TF)");
    add_option("hbfutils-config,u", bpo::value<std::string>()->default_value(std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE)), "config file for HBFUtils (or none)");
    add_option("enable-padding", bpo::value<bool>()->default_value(false)->implicit_value(true), "enable GBT word padding to 128 bits even for RDH V7");
    add_option("ccdb-path", bpo::value<std::string>()->default_value(""), "CCDB url which contains LookupTable");
    add_option("lut-path", bpo::value<std::string>()->default_value(""), "LookupTable path");
    // raw data metric emulation mode
    add_option("emu-rawdata-metrics", bpo::value<uint64_t>()->default_value(0), "Raw data metric emulation, specify 1-byte status word with o2::fit::RawDataMetric::EStatusBits");
    add_option("emu-rawdata-metrics-random", bpo::value<bool>()->default_value(false)->implicit_value(true), "Raw data metric emulation, randomly specifies 1-byte status word with o2::fit::RawDataMetric::EStatusBits. If --emu-rawdata-metrics is not specified then all bits will be on and randomly will be taken");
  }
  void adjustDataFormat()
  { // adjust DataFormat, with or w/o padding
    if (mRdhVersion < 7 && !mEnablePadding) {
      mEnablePadding = true;
      LOG(info) << "padding is always ON for RDH version " << mRdhVersion;
    }
    mDataFormat = mEnablePadding ? 0 : 2;
  }
  // common configs
  int mVerbosity{0};
  std::string mInputFile{"detdigits.root"}; // put det name before "digits"
  std::string mOutputDir{"./"};
  std::string mFlpName{"alio2-cr1-flpnode"}; // put node number after "flp"
  std::string mFileFor{"all"};
  std::string mConfigKeyValues{""};
  // raw data configs
  uint32_t mRdhVersion{o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>()};
  bool mNoEmptyHBF{false};
  std::string mConfDig{std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE)};
  bool mEnablePadding{false}; // will be adjusted(if false) wrt to RDH version
  std::string mCcdbPath{""};
  std::string mChannelMapPath{""};
  // raw data metric emulation
  MetricStatus_t mMetricStatusEmu{}; // status bit for raw data metric emulation
  bool mIsRandomMetric{false};       // use random bits for metric emulation
  // misc field
  uint32_t mDataFormat;
};

/*
 * Wrapper for digit2raw executables
 */
template <typename RawWriterFIT_Type>
struct DigitToRawDevice {
  DigitToRawDevice() = default;
  DigitToRawDevice(const DigitToRawConfig& config) { setConfig(config); }
  RawWriterFIT_Type mRawWriterFIT{};
  DigitToRawConfig mConfig{};
  void setConfig(const DigitToRawConfig& config)
  {
    mConfig = config;
    configure(mConfig);
  }
  void configure(const DigitToRawConfig& cfg)
  {
    // global config
    if (!cfg.mConfDig.empty() && cfg.mConfDig != "none") {
      o2::conf::ConfigurableParam::updateFromFile(cfg.mConfDig, "HBFUtils");
    }
    o2::conf::ConfigurableParam::updateFromString(cfg.mConfigKeyValues);
    // configuring FIT raw writer
    mRawWriterFIT.setFileFor(cfg.mFileFor);
    mRawWriterFIT.setFlpName(cfg.mFlpName);
    mRawWriterFIT.setVerbosity(cfg.mVerbosity);
    if (cfg.mCcdbPath != "") {
      mRawWriterFIT.setCCDBurl(cfg.mCcdbPath);
    }
    if (cfg.mChannelMapPath != "") {
      mRawWriterFIT.setLUTpath(cfg.mChannelMapPath);
    }
    // raw data metric emulation
    mRawWriterFIT.setRandomEmu(cfg.mIsRandomMetric);
    mRawWriterFIT.setStatusEmu(cfg.mMetricStatusEmu);
    // configuring common raw writer
    auto& rawWriter = mRawWriterFIT.getWriter();
    std::string inputGRP = o2::base::NameConf::getGRPFileName();
    const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
    rawWriter.setContinuousReadout(grp->isDetContinuousReadOut(RawWriterFIT_Type::getDetID())); // must be set explicitly
    const int superPageSizeInB = 1024 * 1024;
    rawWriter.setSuperPageSize(superPageSizeInB);
    rawWriter.useRDHVersion(cfg.mRdhVersion);
    rawWriter.setDontFillEmptyHBF(cfg.mNoEmptyHBF);
    rawWriter.useRDHDataFormat(cfg.mDataFormat);
    if (!cfg.mEnablePadding) {        // CRU page alignment padding is used only if no GBT word padding is used
      rawWriter.setAlignmentSize(16); // change to constexpr static field from class?
      rawWriter.setAlignmentPaddingFiller(0xff);
    }
  }
  void run()
  {
    run(mConfig);
  }
  void run(const DigitToRawConfig& cfg)
  {
    // dst path preparations
    o2::raw::assertOutputDirectory(cfg.mOutputDir);
    std::string outDirName(cfg.mOutputDir);
    if (outDirName.back() != '/') {
      outDirName += '/';
    }
    // start timestamp preparations
    const auto& hbfu = o2::raw::HBFUtils::Instance();
    long startTime = hbfu.startTime > 0 ? hbfu.startTime : std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
    // convertation digit->raw
    mRawWriterFIT.convertDigitsToRaw(outDirName, cfg.mInputFile, startTime);
    // config with output info
    auto& rawWriter = mRawWriterFIT.getWriter();
    rawWriter.writeConfFile(rawWriter.getOrigin().str, "RAWDATA", o2::utils::Str::concat_string(outDirName, rawWriter.getOrigin().str, "raw.cfg"));
  }

  static void digit2raw(const DigitToRawConfig& cfg)
  {
    TStopwatch swTot;
    swTot.Start();
    // configurating
    DigitToRawDevice<RawWriterFIT_Type> digitToRawDevice(cfg);
    // running
    digitToRawDevice.run();
    //
    swTot.Stop();
    swTot.Print();
  }
};

} // namespace fit
} // namespace o2

#endif
