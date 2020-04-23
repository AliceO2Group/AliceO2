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
#include "DetectorsRaw/RDHUtils.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "DetectorsRaw/SimpleRawReader.h"
#include "DetectorsRaw/SimpleSTF.h"
#include "CommonConstants/Triggers.h"
#include "Framework/Logger.h"
#include "Framework/InputRecord.h"
#include "DPLUtils/DPLRawParser.h"

// @brief test and demo for RawFileReader and Writer classes
// @author ruben.shahoyan@cern.ch

namespace o2
{
using namespace o2::raw;
using namespace o2::framework;
using RDH = o2::raw::RawFileWriter::RDH;
using IR = o2::InteractionRecord;

constexpr int NCRU = 3 + 1;    // number of CRUs, the last one is a special CRU with preformatted data filled
constexpr int NLinkPerCRU = 4; // number of links per CRU
// sizes for preformatted pages filling (RDH size will be subtracted from the payload) in the last special CRU
constexpr std::array<int, NLinkPerCRU> SpecSize = {512, 1024, 8192, 8192};
constexpr int NPreformHBFPerTF = 32; // number of HBFs with preformatted input per HBF for special last CRU
const std::string PLHeader = "HEADER          ";
const std::string PLTrailer = "TRAILER         ";
const std::string HBFEmpty = "EMPTY_HBF       ";
const std::string CFGName = "test_RawReadWrite_.cfg";

int nPreformatPages = 0;

//
// ========================= simple detector data writer ================================
//
struct TestRawWriter { // simple class to create detector payload for multiple links

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
        auto& link = writer.registerLink((icru << 8) + il, icru, il, 0, outFileName);
        link.rdhCopy.detectorField = 0xff << icru; // if needed, set extra link info, will be copied to all RDHs
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
      for (int icru = 0; icru < NCRU - 1; icru++) {
        // we will create non-0 payload for all but 1st link of every CRU, the writer should take care
        // of creating empty HBFs for the links w/o data
        for (int il = 0; il < NLinkPerCRU; il++) {
          buffer.clear();
          int nGBT = gRandom->Poisson(RDHUtils::MAXCRUPage / RDHUtils::GBTWord * (il));
          if (nGBT) {
            buffer.resize((nGBT + 2) * RDHUtils::GBTWord, icru * NLinkPerCRU + il); // reserve 16B words accounting for the Header and Trailer
            std::memcpy(buffer.data(), PLHeader.c_str(), RDHUtils::GBTWord);
            std::memcpy(buffer.data() + buffer.size() - RDHUtils::GBTWord, PLTrailer.c_str(), RDHUtils::GBTWord);
            // we don't care here about the content of the payload, except the presence of header and trailer
          }
          writer.addData((icru << 8) + il, icru, il, 0, ir, buffer);
        }
      }
    }
    // fill special CRU with preformatted pages
    auto irHB = HBFUtils::Instance().getFirstIR(); // IR of the TF0/HBF0
    int cruID = NCRU - 1;
    while (irHB < irs.back()) {
      for (int il = 0; il < NLinkPerCRU; il++) {
        buffer.clear();
        int pgSize = SpecSize[il] - sizeof(RDH);
        buffer.resize(pgSize);
        for (int ipg = 2 * (NLinkPerCRU - il); ipg--;) {                       // just to enforce writing multiple pages per selected HBFs
          writer.addData((cruID << 8) + il, cruID, il, 0, irHB, buffer, true); // last argument is there to enforce a special "preformatted" mode
          nPreformatPages++;
        }
      }
      irHB.orbit += HBFUtils::Instance().getNOrbitsPerTF() / NPreformHBFPerTF; // we will write 32 such HBFs per TF
    }

    writer.writeConfFile("FLP", "RAWDATA", CFGName); // for further use we write the configuration file
    writer.close(); // flush buffers and close outputs
  }

  // optional callback functions to register in the RawFileWriter
  //_________________________________________________________________
  void emptyHBFMethod(const RDH& rdh, std::vector<char>& toAdd) const
  {
    // what we want to add for every empty page
    toAdd.resize(RDHUtils::GBTWord);
    std::memcpy(toAdd.data(), HBFEmpty.c_str(), RDHUtils::GBTWord);
  }

  //_________________________________________________________________
  int carryOverMethod(const RDH& rdh, const gsl::span<char> data, const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const
  {
    // how we want to split the large payloads. The data is the full payload which was sent for writing and
    // it is already equiped with header and trailer
    if (maxSize <= RDHUtils::GBTWord) { // do not carry over trailer or header only
      return 0;
    }

    // here we simply copy the header/trailer of the payload to every CRU page of this payload
    header.resize(RDHUtils::GBTWord);
    std::memcpy(header.data(), &data[0], RDHUtils::GBTWord);
    trailer.resize(RDHUtils::GBTWord);
    std::memcpy(trailer.data(), &data[data.size() - RDHUtils::GBTWord], RDHUtils::GBTWord);
    // since we write an extra GBT word (trailer) in the end of the CRU page, we ask to write
    // not the block ptr : ptr+maxSize, but ptr : ptr+maxSize - GBTWord;
    int sz = maxSize - RDHUtils::GBTWord;
    return sz;
  }
};

struct TestRawReader { // simple class to read detector raw data for multiple links

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
    testStr.resize(RDHUtils::GBTWord);
    buffers.resize(nLinks); // 1 buffer per link
    firstHBF.resize(nLinks, true);

    int nLinksRead = 0, nPreformatRead = 0;
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

            BOOST_CHECK(RDHUtils::checkRDH(rdhi));                             // check RDH validity
            BOOST_CHECK(RDHUtils::getHeartBeatIR(rdhRef) == RDHUtils::getHeartBeatIR(rdhi)); // make sure the RDH of each link corresponds to the same BC
            if (rdhi.stop) {                                                   // closing page must be empty
              BOOST_CHECK(rdhi.memorySize == rdhi.headerSize);
            } else {
              if (rdhi.cruID < NCRU - 1) {                                    // these are not special CRUs
                BOOST_CHECK(rdhi.memorySize > rdhi.headerSize);               // in this model all non-closing pages must contain something
                if (rdhi.memorySize - rdhi.headerSize == RDHUtils::GBTWord) { // empty HBF will contain just a status word
                  testStr.assign(ptr + rdhi.headerSize, RDHUtils::GBTWord);
                  BOOST_CHECK(testStr == HBFEmpty);
                } else {
                  // pages with real payload should have at least header + trailer + some payload
                  BOOST_CHECK(rdhi.memorySize - rdhi.headerSize > 2 * RDHUtils::GBTWord);
                  testStr.assign(ptr + rdhi.headerSize, RDHUtils::GBTWord);
                  BOOST_CHECK(testStr == PLHeader);
                  testStr.assign(ptr + rdhi.memorySize - RDHUtils::GBTWord, RDHUtils::GBTWord);
                  BOOST_CHECK(testStr == PLTrailer);
                }
              } else { // for the special CRU with preformatted data make sure the page sizes were not modified
                if (rdhi.memorySize > sizeof(RDH) + RDHUtils::GBTWord) {
                  auto tfhb = HBFUtils::Instance().getTFandHBinTF({RDHUtils::getHeartBeatBC(rdhi), RDHUtils::getHeartBeatOrbit(rdhi)}); // TF and HBF relative to TF
                  BOOST_CHECK(tfhb.second % (HBFUtils::Instance().getNOrbitsPerTF() / NPreformHBFPerTF) == 0);            // we were filling only every NPreformHBFPerTF-th HBF
                  BOOST_CHECK(rdhi.memorySize == SpecSize[rdhi.linkID]);                                                  // check if the size is correct
                  nPreformatRead++;
                }
              }
            }
            ptr += rdhi.offsetToNext;
          }
          firstHBF[il] = false;
        }
      }
    } while (nLinksRead); // read until there is something to read

    BOOST_CHECK(nPreformatRead == nPreformatPages); // make sure no preformatted page was lost

  } // run
};

BOOST_AUTO_TEST_CASE(RawReaderWriter)
{
  TestRawWriter dw;
  dw.init();
  dw.run(); // write output
  //
  TestRawReader dr;
  dr.init();
  dr.run(); // read back and check

  // test SimpleReader
  int nLoops = 5;
  SimpleRawReader sr(CFGName, false, nLoops);
  int ntf = 0;
  while (sr.loadNextTF()) {
    ntf++;
    auto& record = *sr.getInputRecord();
    BOOST_CHECK(record.size() == NCRU * NLinkPerCRU);
    o2::header::DataHeader const* dhPrev = nullptr;
    DPLRawParser parser(record);
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      auto const* rdh = it.get_if<RDH>();
      auto const* dh = it.o2DataHeader();
      BOOST_REQUIRE(rdh != nullptr);
      bool newLink = false;
      if (dh != dhPrev) {
        dhPrev = dh;
        newLink = true;
      }
      if (rdh->cruID == NCRU - 1) {
        if (newLink) {
          dh->print();
        }
        RDHUtils::printRDH(rdh);
        if (rdh->memorySize > sizeof(RDH) + RDHUtils::GBTWord) { // special CRU with predefined sizes
          BOOST_CHECK(it.size() + sizeof(RDH) == SpecSize[rdh->linkID]);
        }
      }
    }
  }
}

} // namespace o2
