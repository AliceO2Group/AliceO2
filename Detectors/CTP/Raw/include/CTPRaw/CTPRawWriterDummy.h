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
#include "CommonUtils/StringUtils.h"

namespace o2
{
namespace ctp
{
//
// ========================= simple detector data writer ================================
//
struct CTPRawWriterDummy { // simple class to create detector payload for multiple links

  RawFileWriter writer{"CTP"};
  std::string configName = "rawConf.cfg";

  //_________________________________________________________________
  CTPTestRawWriter(o2::header::DataOrigin origin = "TST", bool isCRU = true, const std::string& cfg = "rawConf.cfg") : writer(origin, isCRU), configName(cfg) {}

  //_________________________________________________________________
  void init()
  {
    // init writer
    writer.useRDHVersion(6);
    int feeIDShift = writer.isCRUDetector() ? 8 : 9;
    // register links
    for (int icru = 0; icru < NCRU; icru++) {
      std::string outFileName = o2::utils::concat_string("testdata_", writer.isCRUDetector() ? "cru" : "rorc", std::to_string(icru), ".raw");
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
          //int nGBT = gRandom->Poisson(RDHUtils::MAXCRUPage / RDHUtils::GBTWord * (il));
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
    writer.writ//
            // ========================= simple detector data writer ================================
            //
            struct CTPTestRawWriter { // simple class to create detector payload for multiple links

              RawFileWriter writer{"CTP"};
              std::string configName = "rawConf.cfg";

              //_________________________________________________________________
              CTPTestRawWriter(o2::header::DataOrigin origin = "TST", bool isCRU = true, const std::string& cfg = "rawConf.cfg") : writer(origin, isCRU), configName(cfg) {}

              //_________________________________________________________________
              void init()
              {
                // init writer
                writer.useRDHVersion(6);
                int feeIDShift = writer.isCRUDetector() ? 8 : 9;
                // register links
                for (int icru = 0; icru < NCRU; icru++) {
                  std::string outFileName = o2::utils::concat_string("testdata_", writer.isCRUDetector() ? "cru" : "rorc", std::to_string(icru), ".raw");
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
                      //int nGBT = gRandom->Poisson(RDHUtils::MAXCRUPage / RDHUtils::GBTWord * (il));
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
eConfFile(writer.getOrigin().str, "RAWDATA", configName);
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
}
}

