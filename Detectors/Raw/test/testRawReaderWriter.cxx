// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test RawReaderWriter class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <TRandom.h>
#include <boost/test/unit_test.hpp>
#include "Steer/InteractionSampler.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "DetectorsRaw/RawFileReader.h"
#include "CommonConstants/Triggers.h"
#include "Framework/Logger.h"

// @brief test and demo for RawFileReader and Writer classes
// @author ruben.shahoyan@cern.ch

namespace o2
{
using namespace o2::raw;
using RDH = o2::header::RAWDataHeaderV4;
using IR = o2::InteractionRecord;

constexpr int NCRU = 3;        // number of CRUs
constexpr int NLinkPerCRU = 4; // number of links per CRU
const std::string PLHeader = "HEADER          ";
const std::string PLTrailer = "TRAILER         ";
const std::string HBFEmpty = "EMPTY_HBF       ";
const std::string CFGName = "test_RawReadWrite_.cfg";

//
// ========================= simple detector data writer ================================
//
struct SimpleRawWriter { // simple class to create detector payload for multiple links

  // suppose detector puts in front and end of every trigger payload some header and trailer

  RawFileWriter writer;

  //_________________________________________________________________
  void init()
  {
    // init writer

    // register links
    for (int icru = 0; icru < NCRU; icru++) {
      std::string outFileName = "testdata_cru" + std::to_string(icru) + ".raw";
      for (int il = 0; il < NLinkPerCRU; il++) {
        writer.registerLink((icru << 8) + il, icru, il, 0, outFileName);
      }
    }

    writer.setContinuousReadout();     // in case we want to issue StartOfContinuous trigger in the beginning
    writer.setCarryOverCallBack(this); // we want that writer to ask the detector code how to split large payloads
    writer.setEmptyPageCallBack(this); // we want the writer to ask the detector code what to put in empty HBFs
  }

  //_________________________________________________________________
  void run()
  {
    // write payload and close outputs

    // generate interaction records for triggers to write
    std::vector<o2::InteractionTimeRecord> irs(1000);
    o2::steer::InteractionSampler irSampler;
    irSampler.setInteractionRate(12000); // ~1.5 interactions per orbit
    irSampler.init();
    irSampler.generateCollisionTimes(irs);

    std::vector<char> buffer;

    // create payload for every interaction and push it to writer
    for (const auto& ir : irs) {
      for (int icru = 0; icru < NCRU; icru++) {
        // we will create non-0 payload for all but 1st link of every CRU, the writer should take care
        // of creating empty HBFs for the links w/o data
        for (int il = 0; il < NLinkPerCRU; il++) {
          int nGBT = gRandom->Poisson(HBFUtils::MAXCRUPage / HBFUtils::GBTWord * (il));
          if (nGBT) {
            buffer.resize((nGBT + 2) * HBFUtils::GBTWord); // reserve 16B words accounting for the Header and Trailer
            std::memcpy(buffer.data(), PLHeader.c_str(), HBFUtils::GBTWord);
            std::memcpy(buffer.data() + buffer.size() - HBFUtils::GBTWord, PLTrailer.c_str(), HBFUtils::GBTWord);
            // we don't care here about the content of the payload, except the presence of header and trailer
          } else {
            buffer.clear();
          }
          writer.addData(icru, il, 0, ir, buffer);
        }
      }
    }

    // for further use we write the configuration file for the output
    {
      std::ofstream cfgfile;
      cfgfile.open(CFGName);
      cfgfile << "[defaults]" << std::endl;
      cfgfile << "dataOrigin = FLP" << std::endl;
      cfgfile << "dataDescription = RAWDATA" << std::endl;
      for (int i = 0; i < writer.getNOutputFiles(); i++) {
        cfgfile << "[input-cru" << i << "]" << std::endl;
        cfgfile << "filePath = " << writer.getOutputFileName(i) << std::endl;
      }
      cfgfile.close();
    }

    writer.close(); // flush buffers and close outputs
  }

  // optional callback functions to register in the RawFileWriter
  //_________________________________________________________________
  void emptyHBFMethod(const RDH& rdh, std::vector<char>& toAdd) const
  {
    // what we want to add for every empty page
    toAdd.resize(HBFUtils::GBTWord);
    std::memcpy(toAdd.data(), HBFEmpty.c_str(), HBFUtils::GBTWord);
  }

  //_________________________________________________________________
  int carryOverMethod(const RDH& rdh, const gsl::span<char> data, const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const
  {
    // how we want to split the large payloads. The data is the full payload which was sent for writing and
    // it is already equiped with header and trailer
    if (maxSize <= HBFUtils::GBTWord) { // do not carry over trailer or header only
      return 0;
    }

    // here we simply copy the header/trailer of the payload to every CRU page of this payload
    header.resize(HBFUtils::GBTWord);
    std::memcpy(header.data(), &data[0], HBFUtils::GBTWord);
    trailer.resize(HBFUtils::GBTWord);
    std::memcpy(trailer.data(), &data[data.size() - HBFUtils::GBTWord], HBFUtils::GBTWord);
    // since we write an extra GBT word (trailer) in the end of the CRU page, we ask to write
    // not the block ptr : ptr+maxSize, but ptr : ptr+maxSize - GBTWord;
    int sz = maxSize - HBFUtils::GBTWord;
    return sz;
  }
};

struct SimpleRawReader { // simple class to read detector raw data for multiple links

  std::unique_ptr<RawFileReader> reader;

  //_________________________________________________________________
  void init()
  {
    reader = std::make_unique<RawFileReader>(CFGName); // init from configuration file
    uint32_t errCheck = 0xffffffff;
    errCheck ^= 0x1 << RawFileReader::ErrNoSuperPageForTF; // makes no sense for superpages not interleaved by others
    reader->setCheckErrors(errCheck);
    reader->init();
  }

  //_________________________________________________________________
  void run()
  {
    // read data of all links
    if (!reader) {
      init();
    }
    int nLinks = reader->getNLinks();
    BOOST_CHECK(nLinks == NCRU * NLinkPerCRU);

    // make sure no errors detected after initialization
    for (int il = 0; il < nLinks; il++) {
      auto& lnk = reader->getLink(il);
      BOOST_CHECK(lnk.nErrors == 0);
    }

    std::vector<std::vector<char>> buffers;
    std::vector<bool> firstHBF;
    std::string testStr;
    testStr.resize(HBFUtils::GBTWord);
    buffers.resize(nLinks); // 1 buffer per link
    firstHBF.resize(nLinks, true);

    int nLinksRead = 0;
    do {
      nLinksRead = 0;
      for (int il = 0; il < nLinks; il++) {
        auto& buff = buffers[il];
        buff.clear();
        auto& lnk = reader->getLink(il);
        auto sz = lnk.getNextHBFSize();
        if (!sz) {
          continue;
        }
        buff.resize(sz);
        BOOST_CHECK(lnk.readNextHBF(buff.data()) == sz);
        nLinksRead++;
      }
      if (nLinksRead) {
        BOOST_CHECK(nLinksRead == nLinks); // all links should have the same number of HBFs

        const auto rdhRef = *reinterpret_cast<RDH*>(buffers[0].data());

        for (int il = 0; il < nLinks; il++) {
          auto& buff = buffers[il];
          int hbsize = buff.size();
          char* ptr = buff.data();
          while (ptr < &buff.back()) { // check all RDH open/close and optional headers and trailers
            const auto rdhi = *reinterpret_cast<RDH*>(ptr);
            if (firstHBF[il]) { // make sure SOT or SOC is there
              BOOST_CHECK(rdhi.triggerType & (o2::trigger::SOC | o2::trigger::SOT));
            }

            BOOST_CHECK(HBFUtils::checkRDH(rdhi));                             // check RDH validity
            BOOST_CHECK(HBFUtils::getHBIR(rdhRef) == HBFUtils::getHBIR(rdhi)); // make sure the RDH of each link corresponds to the same BC
            if (rdhi.stop) {                                                   // closing page must be empty
              BOOST_CHECK(rdhi.memorySize == rdhi.headerSize);
            } else {
              BOOST_CHECK(rdhi.memorySize > rdhi.headerSize);               // in this model all non-closing pages must contain something
              if (rdhi.memorySize - rdhi.headerSize == HBFUtils::GBTWord) { // empty HBF will contain just a status word
                testStr.assign(ptr + rdhi.headerSize, HBFUtils::GBTWord);
                BOOST_CHECK(testStr == HBFEmpty);
              } else {
                // pages with real payload should have at least header + trailer + some payload
                BOOST_CHECK(rdhi.memorySize - rdhi.headerSize > 2 * HBFUtils::GBTWord);
                testStr.assign(ptr + rdhi.headerSize, HBFUtils::GBTWord);
                BOOST_CHECK(testStr == PLHeader);
                testStr.assign(ptr + rdhi.memorySize - HBFUtils::GBTWord, HBFUtils::GBTWord);
                BOOST_CHECK(testStr == PLTrailer);
              }
            }
            ptr += rdhi.offsetToNext;
          }
          firstHBF[il] = false;
        }
      }
    } while (nLinksRead); // read until there is something to read
  }                       // run
};

BOOST_AUTO_TEST_CASE(RawReaderWriter)
{
  SimpleRawWriter dw;
  dw.init();
  dw.run(); // write output
  //
  SimpleRawReader dr;
  dr.init();
  dr.run(); // read back and check
}

} // namespace o2
