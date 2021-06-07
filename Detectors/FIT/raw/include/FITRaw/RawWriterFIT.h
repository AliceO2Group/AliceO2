// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//file RawWriterFIT.h Base class for RAW data writing
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

namespace o2
{
namespace fit
{

// Raw writer for FIT, based on RawReader
template <typename DigitBlockFITtype, typename DataBlockPMtype, typename DataBlockTCMtype>
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
  void setFilePerLink(bool makeFilePerLink) { mOutputPerLink = makeFilePerLink; }
  bool getFilePerLink() const { return mOutputPerLink; }
  void setVerbosity(int verbosityLevel) { mVerbosity = verbosityLevel; }
  int getVerbosity() const { return mVerbosity; }
  int carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                      const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const
  {
    return 0; // do not split, always start new CRU page
  }
  void convertDigitsToRaw(const std::string& outputDir, const std::string& filenameDigits)
  {
    LOG(INFO) << "Converting Digits to Raw data...";
    mWriter.setCarryOverCallBack(this);
    LookupTable_t::Instance().printFullMap();
    //Preparing topo2FEEmetadata map
    mMapTopo2FEEmetadata.clear();
    mMapTopo2FEEmetadata = LookupTable_t::Instance().template makeMapFEEmetadata<o2::header::RAWDataHeader, RDHUtils>();
    //Preparing filenames
    std::string detNameLowCase = LookupTable_t::sDetectorName;
    std::for_each(detNameLowCase.begin(), detNameLowCase.end(), [](char& c) { c = ::tolower(c); });
    auto makeFilename = [&](const o2::header::RAWDataHeader& rdh) -> std::string {
      std::string maskName = detNameLowCase + "_link";
      std::string outputFilename = mOutputPerLink ? o2::utils::Str::concat_string(outputDir, maskName, std::to_string(RDHUtils::getFEEID(rdh)), ".raw") : o2::utils::Str::concat_string(outputDir, detNameLowCase + ".raw");
      return outputFilename;
    };
    //Registering links
    for (const auto& metadataPair : mMapTopo2FEEmetadata) {
      auto& rdh = metadataPair.second;
      auto outputFilename = makeFilename(rdh);
      mWriter.registerLink(RDHUtils::getFEEID(rdh), RDHUtils::getCRUID(rdh), RDHUtils::getLinkID(rdh), RDHUtils::getEndPointID(rdh), outputFilename);
      LOG(INFO) << "Registering link | "
                << "LinkID: " << static_cast<uint16_t>(RDHUtils::getLinkID(rdh)) << " | EndPointID: " << static_cast<uint16_t>(RDHUtils::getEndPointID(rdh)) << " | Output filename: " << outputFilename;
    }
    //Processing digits into raw data
    TFile* inputFile = TFile::Open(filenameDigits.c_str());
    assert(inputFile != nullptr);
    LOG(INFO) << "Source file: " << filenameDigits;
    TTree* inputTree = dynamic_cast<TTree*>(inputFile->Get("o2sim"));
    DigitBlockFIT_t::processDigitBlocks(inputTree, *this);
    delete inputTree;
    inputFile->Close();
    delete inputFile;
  }
  void processDigitBlockPerTF(const std::vector<DigitBlockFIT_t>& vecDigitBlock) // Is used in DigitBlockFIT_t::processDigitBlocks for each TF (TTree entry)
  {
    for (const auto& digitBlock : vecDigitBlock) {
      //Processing PM data
      auto mapDataBlockPM = digitBlock.template decomposeDigits<DataBlockPM_t>();
      if (mVerbosity > 0) {
        digitBlock.print();
      }
      for (const auto& dataBlockPair : mapDataBlockPM) {
        const auto& topo = dataBlockPair.first;
        const auto& dataBlock = dataBlockPair.second;
        const auto itRdh = mMapTopo2FEEmetadata.find(topo);
        assert(itRdh != mMapTopo2FEEmetadata.end());
        if (mVerbosity > 0) {
          dataBlock.print();
        }
        const auto& rdh = itRdh->second;
        auto data = dataBlock.serialize();
        mWriter.addData(RDHUtils::getFEEID(rdh), RDHUtils::getCRUID(rdh), RDHUtils::getLinkID(rdh), RDHUtils::getEndPointID(rdh), dataBlock.getInteractionRecord(), data);
      }
      //Processing TCM data
      const auto dataBlockPair = digitBlock.template decomposeDigits<DataBlockTCM_t>();
      const auto& topo = dataBlockPair.first;
      const auto& dataBlock = dataBlockPair.second;
      const auto itRdh = mMapTopo2FEEmetadata.find(topo);
      assert(itRdh != mMapTopo2FEEmetadata.end());
      if (mVerbosity > 0) {
        dataBlock.print();
      }
      const auto& rdh = itRdh->second;
      auto data = dataBlock.serialize();
      mWriter.addData(RDHUtils::getFEEID(rdh), RDHUtils::getCRUID(rdh), RDHUtils::getLinkID(rdh), RDHUtils::getEndPointID(rdh), dataBlock.getInteractionRecord(), data);
    }
  }

  o2::raw::RawFileWriter mWriter{LookupTable_t::sDetectorName};
  std::map<Topo_t, o2::header::RAWDataHeader> mMapTopo2FEEmetadata;
  //const o2::raw::HBFUtils& mSampler = o2::raw::HBFUtils::Instance();
  bool mOutputPerLink = false;
  int mVerbosity = 0;
};
} // namespace fit
} // namespace o2

#endif
