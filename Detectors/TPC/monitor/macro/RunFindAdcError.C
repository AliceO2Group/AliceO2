// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "FairLogger.h"

#include "TPCBase/PadPos.h"
#include "TPCReconstruction/RawReader.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <algorithm>
#include <thread>
#include <mutex>

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

#include "TH2F.h"
#endif

TH2F* hAdcError = nullptr;

bool run_parallel = true;

using namespace std;
using namespace o2::tpc;

struct Result {
  int mRun;
  int mEvent;
  int mRegion;
  int mLink;
  int mSampa;
  int mTimebin;
  int mSyncPos;

  Result() : mRun(-1), mEvent(-1), mRegion(-1), mLink(-1), mSampa(-1), mTimebin(-1), mSyncPos(-1){};
  Result(int run, int event, int region, int link, int sampa, int timebin, int syncpos) : mRun(run), mEvent(event), mRegion(region), mLink(link), mSampa(sampa), mTimebin(timebin), mSyncPos(syncpos){};
};

void loopReader(std::shared_ptr<RawReader> reader_ptr, std::vector<Result>& result_ptr)
{
  auto reader = reader_ptr.get();
  while (reader->loadNextEventNoWrap() >= 0) {
    if (reader->getAdcError()->size() != 0) {
      for (const auto& err : *reader->getAdcError()) {
        result_ptr.emplace_back(
          reader->getRunNumber(),
          reader->getEventNumber(),
          reader->getRegion(),
          reader->getLink(),
          std::get<0>(err),
          std::get<1>(err),
          std::get<2>(err));
      }
    }
  }
  //  LOG(INFO) << reader->getEventInfo(0)->at(0).path << " done" << FairLogger::endl;
}

//__________________________________________________________________________
void RunFindAdcError(int run_min, int run_max)
{
  std::string DATADIR = "/local/data/tpc-beam-test-2017";
  FairLogger* logger = FairLogger::GetLogger();
  logger->SetLogVerbosityLevel("LOW");
  logger->SetLogScreenLevel("INFO");

  // ===========================================================================
  // Preparing File Infos
  // ===========================================================================
  std::vector<std::string> fileInfos;
  for (int r = run_min; r <= run_max; ++r) {
    std::ostringstream ss;
    ss << DATADIR << "/run" << std::setw(6) << std::setfill('0') << r << "/run" << std::setw(6) << std::setfill('0') << r;

    fileInfos.emplace_back(ss.str() + "_trorc00_link00.bin:0:9");
    fileInfos.emplace_back(ss.str() + "_trorc00_link01.bin:1:9");
    //    fileInfos.emplace_back(ss.str()+"_trorc00_link02.bin:0:10");
    //    fileInfos.emplace_back(ss.str()+"_trorc00_link03.bin:1:10");
    //    fileInfos.emplace_back(ss.str()+"_trorc00_link04.bin:0:11");
    //    fileInfos.emplace_back(ss.str()+"_trorc00_link05.bin:1:11");
    //    fileInfos.emplace_back(ss.str()+"_trorc00_link06.bin:2:11");
    //    fileInfos.emplace_back(ss.str()+"_trorc00_link07.bin:3:11");
    //    fileInfos.emplace_back(ss.str()+"_trorc00_link08.bin:2:12");
    //    fileInfos.emplace_back(ss.str()+"_trorc00_link09.bin:3:12");
    fileInfos.emplace_back(ss.str() + "_trorc00_link10.bin:2:13");
    fileInfos.emplace_back(ss.str() + "_trorc00_link11.bin:3:13");
  }

  // ===========================================================================
  // Preparing the Readers
  // ===========================================================================
  LOG(INFO) << "Create all Readers..." << FairLogger::endl;
  std::vector<std::shared_ptr<RawReader>> RawReaders;
  for (const auto& s : fileInfos) {
    auto rawReader = new RawReader;
    rawReader->addInputFile(s);
    rawReader->setUseRawInMode3(true);
    rawReader->setCheckAdcClock(true);

    // keep only raw data
    if (rawReader->getEventInfo(0)->at(0).header.dataType != 2)
      RawReaders.push_back(std::shared_ptr<RawReader>(rawReader));
    else
      LOG(INFO) << "\tDrop Reader of file " << rawReader->getEventInfo(0)->at(0).path << " because of readout mode 2." << FairLogger::endl;
  }
  LOG(INFO) << "... done" << FairLogger::endl;

  // ===========================================================================
  // Loop through the Readers to find ADC errors
  // ===========================================================================
  LOG(INFO) << "Loop through Readers..." << FairLogger::endl;
  std::vector<std::thread> threads;
  std::vector<std::vector<Result>> results(RawReaders.size());
  int i = 0;
  for (auto& reader_ptr : RawReaders) {
    if (run_parallel)
      threads.emplace_back(loopReader, std::ref(reader_ptr), std::ref(results[i]));
    else
      loopReader(std::ref(reader_ptr), std::ref(results[i]));
    ++i;
  }

  for (std::thread& t : threads) {
    t.join();
  }
  LOG(INFO) << "... done" << FairLogger::endl;

  // ===========================================================================
  // Analyse outcome
  // ===========================================================================
  logger->SetLogScreenLevel("DEBUG");
  hAdcError = new TH2F("hAdcError", "occurrence of ADC errors", 36, 0, 36, 20, 0, 20);
  hAdcError->GetXaxis()->SetTitle("SampaID");
  hAdcError->GetYaxis()->SetTitle("Timebin");
  o2::tpc::PadPos padPos;
  std::cout << RawReaders.size() << std::endl;
  for (int i = 0; i < results.size(); ++i) {
    if (results[i].size() == 0)
      continue;
    std::cout << results[i].size() << " " << results[i][0].mEvent << " " << i << std::endl;
    int sampaChip = 0;
    for (const auto& r : results[i]) {
      sampaChip = (r.mRegion * 4) + ((r.mLink - 9) * 3) + r.mSampa;
      hAdcError->Fill(sampaChip, r.mTimebin);
      std::cout
        << r.mRun << " "
        << r.mEvent << " "
        << r.mRegion << " "
        << r.mLink << " "
        << r.mSampa << " "
        << r.mTimebin << " "
        << r.mSyncPos << std::endl;
      RawReaders[i]->loadEvent(r.mEvent);
    }
    //    while (std::shared_ptr<std::vector<uint16_t>> data = RawReaders[i]->getNextData(padPos)) {
    //      if (!data) continue;
    //
    //    }
  }
  hAdcError->Draw("colz");
}
