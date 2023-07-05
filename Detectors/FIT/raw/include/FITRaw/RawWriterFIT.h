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
#include <Rtypes.h>
#include "FITRaw/DataBlockFIT.h"
#include "FITRaw/DigitBlockFIT.h"
#include <Framework/Logger.h>
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "CommonUtils/StringUtils.h"
#include <gsl/span>
#include <fmt/format.h>

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
  using RDHUtils = o2::raw::RDHUtils;
  RawWriterFIT() = default;
  ~RawWriterFIT() = default;
  o2::raw::RawFileWriter& getWriter() { return mWriter; }
  void setFileFor(const std::string& fileFor) { mFileFor = fileFor; }
  void setFlpName(const std::string& flpName) { mFlpName = flpName; }
  bool getFilePerLink() const { return mOutputPerLink; }
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
      // Processing PM data
      auto mapDataBlockPM = digitBlock.template decomposeDigits<DataBlockPM_t>();
      if (mVerbosity > 0) {
        digitBlock.print();
      }
      for (const auto& dataBlockPair : mapDataBlockPM) {
        const auto& topo = dataBlockPair.first;
        const auto& dataBlock = dataBlockPair.second;
        const auto itRdh = mMapTopo2FEEmetadata.find(topo);
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
      const auto dataBlockPair = digitBlock.template decomposeDigits<DataBlockTCM_t>();
      const auto& topo = dataBlockPair.first;
      const auto& dataBlock = dataBlockPair.second;
      const auto itRdh = mMapTopo2FEEmetadata.find(topo);
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

  o2::raw::RawFileWriter mWriter{LookupTable_t::sDetectorName};
  std::string mFlpName{};
  std::string mFileFor{};
  int mDataFormat{0}; // RDH::dataFormat field, 0 - padded, 2 - no padding
  std::map<Topo_t, o2::header::RAWDataHeader> mMapTopo2FEEmetadata;
  // const o2::raw::HBFUtils& mSampler = o2::raw::HBFUtils::Instance();
  bool mOutputPerLink = false;
  int mVerbosity = 0;
};
} // namespace fit
} // namespace o2

#endif
