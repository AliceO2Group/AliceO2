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
#include "Headers/DataHeaderHelpers.h"
#include "DPLUtils/DPLRawParser.h"
#include "CommonUtils/StringUtils.h"

// @brief test and demo for RawFileReader and Writer classes
// @author ruben.shahoyan@cern.ch

namespace o2
{
using namespace o2::raw;
using namespace o2::framework;
using IR = o2::InteractionRecord;
using RDHAny = header::RDHAny;
constexpr int NCRU = 3 + 1;    // number of CRUs, the last one is a special CRU with preformatted data filled
constexpr int NLinkPerCRU = 4; // number of links per CRU
// sizes for preformatted pages filling (RDH size will be subtracted from the payload) in the last special CRU
constexpr std::array<int, NLinkPerCRU> SpecSize = {512, 1024, 8192, 8192};
constexpr int NPreformHBFPerTF = 32; // number of HBFs with preformatted input per HBF for special last CRU
const std::string PLHeader = "HEADER          ";
const std::string PLTrailer = "TRAILER         ";
const std::string HBFEmpty = "EMPTY_HBF       ";
const std::string CFGName = "testRawReadWrite";

int nPreformatPages = 0;

//
// ========================= simple detector data writer ================================
//
struct TestRawWriter { // simple class to create detector payload for multiple links

  RawFileWriter writer{"TST"};
  std::string configName = "rawConf.cfg";

  //_________________________________________________________________
  TestRawWriter(o2::header::DataOrigin origin = "TST", bool isCRU = true, const std::string& cfg = "rawConf.cfg") : writer(origin, isCRU), configName(cfg) {}

  //_________________________________________________________________
  void init()
  {
    // init writer
    writer.useRDHVersion(6);
    int feeIDShift = writer.isCRUDetector() ? 8 : 9;
    // register links
    for (int icru = 0; icru < NCRU; icru++) {
      std::string outFileName = o2::utils::Str::concat_string("testdata_", writer.isCRUDetector() ? "cru" : "rorc", std::to_string(icru), ".raw");
      for (int il = 0; il < NLinkPerCRU; il++) {
        auto& link = writer.registerLink((icru << feeIDShift) + il, icru, il, 0, outFileName);
        RDHUtils::setDetectorField(link.rdhCopy, 0xff << icru); // if needed, set extra link info, will be copied to all RDHs
      }
    }

    if (writer.isCRUDetector()) {
      writer.setContinuousReadout();     // in case we want to issue StartOfContinuous trigger in the beginning
      writer.setEmptyPageCallBack(this); // we want the writer to ask the detector code what to put in empty HBFs
    }
    writer.setCarryOverCallBack(this); // we want that writer to ask the detector code how to split large payloads

    writer.setApplyCarryOverToLastPage(true); // call CarryOver method also for the last chunk
    writer.doLazinessCheck(false);            // do not apply auto-completion since the test mixes preformatted links filled per HBF and standard links filled per IR.
  }

  //_________________________________________________________________
  void run()
  {
    // write payload and close outputs
    nPreformatPages = 0;
    // generate interaction records for triggers to write
    std::vector<o2::InteractionTimeRecord> irs(1000);
    o2::steer::InteractionSampler irSampler;
    irSampler.setInteractionRate(12000); // ~1.5 interactions per orbit
    irSampler.init();
    irSampler.generateCollisionTimes(irs);

    std::vector<char> buffer;
    int feeIDShift = writer.isCRUDetector() ? 8 : 9;

    // create payload for every interaction and push it to writer
    for (const auto& ir : irs) {
      int nCRU2Fill = writer.isCRUDetector() ? NCRU - 1 : NCRU; // in CRU mode we will fill 1 special CRU with preformatted data
      for (int icru = 0; icru < nCRU2Fill; icru++) {
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
          writer.addData((icru << feeIDShift) + il, icru, il, 0, ir, buffer);
        }
      }
    }
    if (writer.isCRUDetector()) {
      // fill special CRU with preformatted pages
      auto irHB = HBFUtils::Instance().getFirstIR(); // IR of the TF0/HBF0
      int cruID = NCRU - 1;
      while (irHB < irs.back()) {
        for (int il = 0; il < NLinkPerCRU; il++) {
          buffer.clear();
          int pgSize = SpecSize[il] - sizeof(RDHAny);
          buffer.resize(pgSize);
          for (int ipg = 2 * (NLinkPerCRU - il); ipg--;) {                       // just to enforce writing multiple pages per selected HBFs
            writer.addData((cruID << 8) + il, cruID, il, 0, irHB, buffer, true); // last argument is there to enforce a special "preformatted" mode
            nPreformatPages++;
          }
        }
        irHB.orbit += HBFUtils::Instance().getNOrbitsPerTF() / NPreformHBFPerTF; // we will write 32 such HBFs per TF
      }
    }

    // for further use we write the configuration file
    writer.writeConfFile(writer.getOrigin().str, "RAWDATA", configName);
    writer.close(); // flush buffers and close outputs
  }

  // optional callback functions to register in the RawFileWriter
  //_________________________________________________________________
  void emptyHBFMethod(const RDHAny* rdh, std::vector<char>& toAdd) const
  {
    // what we want to add for every empty page
    toAdd.resize(RDHUtils::GBTWord);
    std::memcpy(toAdd.data(), HBFEmpty.c_str(), RDHUtils::GBTWord);
  }

  //_________________________________________________________________
  int carryOverMethod(const RDHAny* rdh, const gsl::span<char> data, const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const
  {
    // how we want to split the large payloads. The data is the full payload which was sent for writing and
    // it is already equiped with header and trailer
    static int verboseCount = 0;

    if (maxSize <= RDHUtils::GBTWord) { // do not carry over trailer or header only
      return 0;
    }

    int bytesLeft = data.size() - (ptr - &data[0]);
    bool lastPage = bytesLeft <= maxSize;
    if (verboseCount++ < 100) {
      LOG(INFO) << "Carry-over method for chunk of size " << bytesLeft << " is called, MaxSize = " << maxSize << (lastPage ? " : last chunk being processed!" : "");
    }
    // here we simply copy the header/trailer of the payload to every CRU page of this payload
    header.resize(RDHUtils::GBTWord);
    std::memcpy(header.data(), &data[0], RDHUtils::GBTWord);
    trailer.resize(RDHUtils::GBTWord);
    std::memcpy(trailer.data(), &data[data.size() - RDHUtils::GBTWord], RDHUtils::GBTWord);
    // since we write an extra GBT word (trailer) in the end of the CRU page, we ask to write
    // not the block ptr : ptr+maxSize, but ptr : ptr+maxSize - GBTWord;
    int sz = maxSize; // if the method is called for the last page, then the trailer is overwritten !!!
    if (!lastPage) {  // otherwise it is added incrementally, so its size must be accounted
      sz -= trailer.size();
    }
    return sz;
  }
};

struct TestRawReader { // simple class to read detector raw data for multiple links

  std::unique_ptr<RawFileReader> reader;
  std::string confName;

  //_________________________________________________________________
  TestRawReader(const std::string& name = "TST", const std::string& cfg = "rawConf.cfg") : confName(cfg) {}

  //_________________________________________________________________
  void init()
  {
    reader = std::make_unique<RawFileReader>(confName); // init from configuration file
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
        auto sz = lnk.getNextHBFSize(); // HBF treated as a trigger for RORC detectors
        if (!sz) {
          continue;
        }
        buff.resize(sz);
        BOOST_CHECK(lnk.readNextHBF(buff.data()) == sz);
        nLinksRead++;
      }
      if (nLinksRead) {
        BOOST_CHECK(nLinksRead == nLinks); // all links should have the same number of HBFs or triggers

        const auto rdhRef = *reinterpret_cast<RDHAny*>(buffers[0].data());

        for (int il = 0; il < nLinks; il++) {
          auto& lnk = reader->getLink(il);
          auto& buff = buffers[il];
          int hbsize = buff.size();
          char* ptr = buff.data();
          while (ptr < &buff.back()) { // check all RDH open/close and optional headers and trailers
            const auto rdhi = *reinterpret_cast<RDHAny*>(ptr);
            if (firstHBF[il]) { // make sure SOT or SOC is there
              BOOST_CHECK(RDHUtils::getTriggerType(rdhi) & (o2::trigger::SOC | o2::trigger::SOT));
            }
            auto memSize = RDHUtils::getMemorySize(rdhi);
            auto rdhSize = RDHUtils::getHeaderSize(rdhi);
            BOOST_CHECK(RDHUtils::checkRDH(rdhi));                             // check RDH validity

            if (!(RDHUtils::getHeartBeatIR(rdhRef) == RDHUtils::getHeartBeatIR(rdhi))) {
              RDHUtils::printRDH(rdhRef);
              RDHUtils::printRDH(rdhi);
            }

            BOOST_CHECK(RDHUtils::getHeartBeatIR(rdhRef) == RDHUtils::getHeartBeatIR(rdhi)); // make sure the RDH of each link corresponds to the same BC
            if (RDHUtils::getStop(rdhi)) {                                                   // closing page must be empty
              BOOST_CHECK(memSize == rdhSize);
            } else {
              if (!lnk.cruDetector || RDHUtils::getCRUID(rdhi) < NCRU - 1) { // only last CRU of in non-RORC mode was special
                if (lnk.cruDetector) {
                  BOOST_CHECK(memSize > rdhSize); // in this model all non-closing pages must contain something
                }
                if (memSize - rdhSize == RDHUtils::GBTWord) { // empty HBF will contain just a status word
                  if (lnk.cruDetector) {
                    testStr.assign(ptr + rdhSize, RDHUtils::GBTWord);
                    BOOST_CHECK(testStr == HBFEmpty);
                  }
                } else if (memSize > rdhSize) {
                  // pages with real payload should have at least header + trailer + some payload
                  BOOST_CHECK(memSize - rdhSize > 2 * RDHUtils::GBTWord);
                  testStr.assign(ptr + rdhSize, RDHUtils::GBTWord);
                  BOOST_CHECK(testStr == PLHeader);
                  testStr.assign(ptr + memSize - RDHUtils::GBTWord, RDHUtils::GBTWord);
                  BOOST_CHECK(testStr == PLTrailer);
                }
              } else { // for the special CRU with preformatted data make sure the page sizes were not modified
                if (memSize > rdhSize + RDHUtils::GBTWord) {
                  auto tfhb = HBFUtils::Instance().getTFandHBinTF({RDHUtils::getHeartBeatBC(rdhi), RDHUtils::getHeartBeatOrbit(rdhi)}); // TF and HBF relative to TF
                  BOOST_CHECK(tfhb.second % (HBFUtils::Instance().getNOrbitsPerTF() / NPreformHBFPerTF) == 0);            // we were filling only every NPreformHBFPerTF-th HBF
                  BOOST_CHECK(memSize == SpecSize[RDHUtils::getLinkID(rdhi)]);                                            // check if the size is correct
                  nPreformatRead++;
                }
              }
            }
            ptr += RDHUtils::getOffsetToNext(rdhi);
          }
          firstHBF[il] = false;
        }
      }
    } while (nLinksRead); // read until there is something to read

    BOOST_CHECK(nPreformatRead == nPreformatPages); // make sure no preformatted page was lost

  } // run
};

BOOST_AUTO_TEST_CASE(RawReaderWriter_CRU)
{
  TestRawWriter dw{"TST", true, "test_raw_conf_GBT.cfg"}; // this is a CRU detector with origin TST
  dw.init();
  dw.run(); // write output
  //
  TestRawReader dr{"TST", "test_raw_conf_GBT.cfg"}; // here we set the reader wrapper name just to deduce the input config name, everything else will be deduced from the config
  dr.init();
  dr.run(); // read back and check

  // test SimpleReader
  int nLoops = 5;
  SimpleRawReader sr(dr.confName, false, nLoops);
  int ntf = 0;
  while (sr.loadNextTF()) {
    ntf++;
    auto& record = *sr.getInputRecord();
    BOOST_CHECK(record.size() == NCRU * NLinkPerCRU);
    o2::header::DataHeader const* dhPrev = nullptr;
    DPLRawParser parser(record);
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      //      auto const* rdh = &get_if<RDHAny>();
      auto const* rdh = reinterpret_cast<const RDHAny*>(it.raw()); // RSTODO this is a hack in absence of generic header getter
      auto const* dh = it.o2DataHeader();
      BOOST_REQUIRE(rdh != nullptr);
      bool newLink = false;
      if (dh != dhPrev) {
        dhPrev = dh;
        newLink = true;
      }
      if (RDHUtils::getCRUID(*rdh) == NCRU - 1) {
        if (newLink) {
          LOGP(INFO, "{}", *dh);
        }
        RDHUtils::printRDH(rdh);
        if (RDHUtils::getMemorySize(*rdh) > sizeof(RDHAny) + RDHUtils::GBTWord) { // special CRU with predefined sizes
          BOOST_CHECK(it.size() + sizeof(RDHAny) == SpecSize[RDHUtils::getLinkID(*rdh)]);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(RawReaderWriter_RORC)
{
  TestRawWriter dw{"TST", false, "test_raw_conf_DDL.cfg"}; // this is RORC detector with origin TST
  dw.init();
  dw.run(); // write output
  //
  TestRawReader dr{"TST", "test_raw_conf_DDL.cfg"}; // here we set the reader wrapper name just to deduce the input config name, everything else will be deduced from the config
  dr.init();
  dr.run(); // read back and check

  // test SimpleReader
  int nLoops = 5;
  SimpleRawReader sr(dr.confName, false, nLoops);
  int ntf = 0;
  while (sr.loadNextTF()) {
    ntf++;
    auto& record = *sr.getInputRecord();
    LOG(INFO) << "FAIL? " << record.size() << " " << NCRU * NLinkPerCRU;

    BOOST_CHECK(record.size() == NCRU * NLinkPerCRU);
    o2::header::DataHeader const* dhPrev = nullptr;
    DPLRawParser parser(record);
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      //      auto const* rdh = &get_if<RDHAny>();
      auto const* rdh = reinterpret_cast<const RDHAny*>(it.raw()); // RSTODO this is a hack in absence of generic header getter
      auto const* dh = it.o2DataHeader();
      BOOST_REQUIRE(rdh != nullptr);
      bool newLink = false;
      if (dh != dhPrev) {
        dhPrev = dh;
        newLink = true;
      }
    }
  }
}

} // namespace o2
