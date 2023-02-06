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

/// \file rawReaderFileNew.cxx
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory

#include <bitset>
#include <iostream>
#include <boost/program_options.hpp>
#include <gsl/span>
#include <fairlogger/Logger.h>

#include <TFile.h>
#include <TTree.h>

#include "CommonConstants/Triggers.h"
#include "DetectorsRaw/RawFileReader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "FOCALReconstruction/PadWord.h"
#include "FOCALReconstruction/PadDecoder.h"
#include "Headers/RDHAny.h"

namespace bpo = boost::program_options;

// Data for tree
struct PadTreeData {
  static constexpr int NASICS = 20;
  static constexpr int NCHANNELS = 72;
  static constexpr int WINDUR = 20;
  static constexpr int NTRIGGER = 8;

  int mBCid;
  int mOrbit;
  int mHeader0[NASICS];
  int mFOURBIT0[NASICS];
  int mWADD0[NASICS];
  int mBCID0[NASICS];
  int mTrailer0[NASICS];
  int mHeader1[NASICS];
  int mFOURBIT1[NASICS];
  int mWADD1[NASICS];
  int mBCID1[NASICS];
  int mTrailer1[NASICS];
  int mASICNum[NASICS];
  int mADC[NASICS][NCHANNELS];
  int mTOA[NASICS][NCHANNELS];
  int mTOT[NASICS][NCHANNELS];
  int mCalib0[NASICS];
  int mCalib1[NASICS];
  int mTriggerhead0[NASICS][WINDUR];
  int mTriggerhead1[NASICS][WINDUR];
  int mTriggerdata[NASICS][NTRIGGER][WINDUR];

  TTree* mTree = nullptr;

  void connectTree(TTree* padtree)
  {
    mTree = padtree;
    mTree->Branch("ORBIT", &mOrbit, "ORBIT/I");
    mTree->Branch("BCID", &mBCid, "BCID/I");
    mTree->Branch("HEAD0", &mHeader0, "HEAD0[20]/I");
    mTree->Branch("FOURBIT0", &mFOURBIT0, "FOURBIT0[20]/I");
    mTree->Branch("BCID0", &mBCID0, "BCID0[20]/I");
    mTree->Branch("WADD0", &mWADD0, "WADD0[20]/I");
    mTree->Branch("TRAILER0", &mTrailer0, "TRAILER0[20]/I");
    mTree->Branch("HEAD1", &mHeader1, "HEAD1[20]/I");
    mTree->Branch("FOURBIT1", &mFOURBIT1, "FOURBIT1[20]/I");
    mTree->Branch("BCID1", &mBCID1, "BCID1[20]/I");
    mTree->Branch("WADD1", &mWADD1, "WADD1[20]/I");
    mTree->Branch("TRAILER1", &mTrailer1, "TRAILER1[20]/I");
    mTree->Branch("ASIC", &mASICNum, "ASICNum[20]/I");
    mTree->Branch("ADC", &mADC, "ADC[20][72]/I");
    mTree->Branch("TOA", &mTOA, "TOA[20][72]/I");
    mTree->Branch("TOT", &mTOT, "TOT[20][72]/I");
    mTree->Branch("CALIB0", &mCalib0, "CALIB0[20]/I");
    mTree->Branch("CALIB1", &mCalib1, "CALIB1[20]/I");
    mTree->Branch("TRIGHEADER0", &mTriggerhead0, "TRIGHEADER0[20][20]/I");
    mTree->Branch("TRIGHEADER1", &mTriggerhead1, "TRIGHEADER1[20][20]/I");
    mTree->Branch("TRIGDATA", &mTriggerdata, "TRIGDATA[20][8][20]/I");
  }

  void reset()
  {
    mBCid = 0;
    mOrbit = 0;
    memset(mHeader0, 0, sizeof(int) * 20);
    memset(mFOURBIT0, 0, sizeof(int) * 20);
    memset(mBCID0, 0, sizeof(int) * 20);
    memset(mWADD0, 0, sizeof(int) * 20);
    memset(mTrailer0, 0, sizeof(int) * 20);
    memset(mHeader1, 0, sizeof(int) * 20);
    memset(mFOURBIT1, 0, sizeof(int) * 20);
    memset(mBCID1, 0, sizeof(int) * 20);
    memset(mWADD1, 0, sizeof(int) * 20);
    memset(mTrailer1, 0, sizeof(int) * 20);
    memset(mASICNum, 0, sizeof(int) * 20);
    memset(mADC, 0, sizeof(int) * 20 * 72);
    memset(mTOA, 0, sizeof(int) * 20 * 72);
    memset(mTOT, 0, sizeof(int) * 20 * 72);
    memset(mCalib0, 0, sizeof(int) * 20);
    memset(mCalib1, 0, sizeof(int) * 20);
    memset(mTriggerhead0, 0, sizeof(int) * 20 * 20);
    memset(mTriggerhead1, 0, sizeof(int) * 20 * 20);
    memset(mTriggerdata, 0, sizeof(int) * 20 * 8 * 20);
  }

  void setInteractionRecord(const o2::InteractionRecord& ir)
  {
    mBCid = ir.bc;
    mOrbit = ir.orbit;
  }

  void fill(const o2::focal::PadData& data)
  {
    for (int iasic = 0; iasic < NASICS; iasic++) {
      auto& asicdata = data.getDataForASIC(iasic);
      auto& asicraw = asicdata.getASIC();
      mASICNum[iasic] = iasic;
      mHeader0[iasic] = asicraw.getFirstHeader().mHeader;
      mFOURBIT0[iasic] = asicraw.getFirstHeader().mFourbit;
      mBCID0[iasic] = asicraw.getFirstHeader().mBCID;
      mWADD0[iasic] = asicraw.getFirstHeader().mWADD;
      mTrailer0[iasic] = asicraw.getFirstHeader().mTrailer;
      mHeader1[iasic] = asicraw.getSecondHeader().mHeader;
      mFOURBIT1[iasic] = asicraw.getSecondHeader().mFourbit;
      mBCID1[iasic] = asicraw.getSecondHeader().mBCID;
      mWADD1[iasic] = asicraw.getSecondHeader().mWADD;
      mTrailer1[iasic] = asicraw.getSecondHeader().mTrailer;
      mCalib0[iasic] = asicraw.getFirstCalib().mADC;
      mCalib1[iasic] = asicraw.getSecondCalib().mADC;
      for (auto ichannel = 0; ichannel < NCHANNELS; ichannel++) {
        mADC[iasic][ichannel] = asicraw.getChannel(ichannel).getADC();
        mTOT[iasic][ichannel] = asicraw.getChannel(ichannel).getTOT();
        mTOA[iasic][ichannel] = asicraw.getChannel(ichannel).getTOA();
      }
      auto triggerdata = asicdata.getTriggerWords();
      for (auto iwin = 0; iwin < WINDUR; iwin++) {
        mTriggerhead0[iasic][iwin] = triggerdata[iwin].mHeader;
        mTriggerhead1[iasic][iwin] = triggerdata[iwin].mHeader;
        mTriggerdata[iasic][0][iwin] = triggerdata[iwin].mTrigger0;
        mTriggerdata[iasic][1][iwin] = triggerdata[iwin].mTrigger1;
        mTriggerdata[iasic][2][iwin] = triggerdata[iwin].mTrigger2;
        mTriggerdata[iasic][3][iwin] = triggerdata[iwin].mTrigger3;
        mTriggerdata[iasic][4][iwin] = triggerdata[iwin].mTrigger4;
        mTriggerdata[iasic][5][iwin] = triggerdata[iwin].mTrigger5;
        mTriggerdata[iasic][6][iwin] = triggerdata[iwin].mTrigger6;
        mTriggerdata[iasic][7][iwin] = triggerdata[iwin].mTrigger7;
      }
    }
  }

  void fillTree()
  {
    mTree->Fill();
  }
};

int convertPadData(gsl::span<const char> padrawdata, const o2::InteractionRecord& currentir, PadTreeData& rootified)
{
  auto payloadsizeGBT = padrawdata.size() * sizeof(char) / sizeof(o2::focal::PadGBTWord);
  auto gbtdata = gsl::span<const o2::focal::PadGBTWord>(reinterpret_cast<const o2::focal::PadGBTWord*>(padrawdata.data()), payloadsizeGBT);
  o2::focal::PadDecoder decoder;

  constexpr std::size_t EVENTSIZEPADGBT = 1180;
  int nevents = gbtdata.size() / EVENTSIZEPADGBT;
  for (int iev = 0; iev < nevents; iev++) {
    decoder.reset();
    rootified.reset();
    decoder.decodeEvent(gbtdata.subspan(iev * EVENTSIZEPADGBT, EVENTSIZEPADGBT));
    rootified.setInteractionRecord(currentir);
    rootified.fill(decoder.getData());
    rootified.fillTree();
  }
  return nevents;
}

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       " <cmds/options>\n"
                                       "  Tool will decode the DDLx data for EMCAL 0\n"
                                       "Commands / Options");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("verbose,v", bpo::value<uint32_t>()->default_value(0), "Select verbosity level [0 = no output]");
    add_option("version", "Print version information");
    add_option("input-file,i", bpo::value<std::string>()->required(), "Specifies input file. Multiple files can be parsed separated by ,");
    add_option("output-file,o", bpo::value<std::string>()->default_value("FOCALPadData.root"), "Output file for rootified data");
    add_option("readout,r", bpo::value<std::string>()->default_value("RORC"), "Readout mode (RORC or CRU)");
    add_option("debug,d", bpo::value<uint32_t>()->default_value(0), "Select debug output level [0 = no debug output]");

    opt_all.add(opt_general).add(opt_hidden);
    bpo::store(bpo::command_line_parser(argc, argv).options(opt_all).positional(opt_pos).run(), vm);

    if (vm.count("help") || argc == 1) {
      std::cout << opt_general << std::endl;
      exit(0);
    }

    if (vm.count("version")) {
      // std::cout << GitInfo();
      exit(0);
    }

    bpo::notify(vm);
  } catch (bpo::error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl
              << std::endl;
    std::cerr << opt_general << std::endl;
    exit(1);
  } catch (std::exception& e) {
    std::cerr << e.what() << ", application will now exit" << std::endl;
    exit(2);
  }

  auto rawfilename = vm["input-file"].as<std::string>();
  auto rootfilename = vm["output-file"].as<std::string>();
  auto readoutmode = vm["readout"].as<std::string>();

  std::vector<std::string> inputfiles;
  if (rawfilename.find(",") != std::string::npos) {
    std::stringstream parser(rawfilename);
    std::string buffer;
    while (std::getline(parser, buffer, ',')) {
      LOG(info) << "Adding " << buffer;
      inputfiles.push_back(buffer);
    }
    LOG(info) << "Found " << inputfiles.size() << " input files to process";
  }

  o2::raw::RawFileReader::ReadoutCardType readout = o2::raw::RawFileReader::RORC;
  if (readoutmode != "RORC" && readoutmode != "CRU") {
    LOG(error) << "Unknown readout mode - select RORC or CRU";
    exit(3);
  } else if (readoutmode == "RORC") {
    LOG(info) << "Reconstructing in C-RORC mode";
    readout = o2::raw::RawFileReader::RORC;
  } else {
    LOG(info) << "Reconstructing in CRU mode";
    readout = o2::raw::RawFileReader::CRU;
  }

  o2::raw::RawFileReader reader;
  reader.setDefaultDataOrigin(o2::header::gDataOriginFOC);
  reader.setDefaultDataDescription(o2::header::gDataDescriptionRawData);
  reader.setDefaultReadoutCardType(readout);
  for (auto rawfile : inputfiles) {
    reader.addFile(rawfile);
  }
  reader.init();

  std::unique_ptr<TFile> rootfilewriter(TFile::Open(rootfilename.data(), "RECREATE"));
  rootfilewriter->cd();
  TTree* padtree = new TTree("PadData", "PadData");
  PadTreeData rootified;
  rootified.connectTree(padtree);

  int nHBFprocessed = 0, nTFprocessed = 0, nEventsProcessed = 0;
  std::map<int, int> nEvnetsHBF;
  while (1) {
    int tfID = reader.getNextTFToRead();
    if (tfID >= reader.getNTimeFrames()) {
      LOG(info) << "nothing left to read after " << tfID << " TFs read";
      break;
    }
    std::vector<char> rawtf; // where to put extracted data
    for (int il = 0; il < reader.getNLinks(); il++) {
      auto& link = reader.getLink(il);

      auto sz = link.getNextTFSize(); // size in bytes needed for the next TF of this link
      rawtf.resize(sz);
      link.readNextTF(rawtf.data());
      gsl::span<char> dataBuffer(rawtf);

      // Parse
      std::vector<char> hbfbuffer;
      int currentpos = 0;
      o2::InteractionRecord currentir;
      while (currentpos < dataBuffer.size()) {
        auto rdh = reinterpret_cast<const o2::header::RDHAny*>(dataBuffer.data() + currentpos);
        o2::raw::RDHUtils::printRDH(rdh);
        if (o2::raw::RDHUtils::getMemorySize(rdh) == o2::raw::RDHUtils::getHeaderSize(rdh)) {
          auto trigger = o2::raw::RDHUtils::getTriggerType(rdh);
          if (trigger & o2::trigger::SOT || trigger & o2::trigger::HB) {
            if (o2::raw::RDHUtils::getStop(rdh)) {
              LOG(debug) << "Stop bit received - processing payload";
              auto nevents = convertPadData(hbfbuffer, currentir, rootified);
              hbfbuffer.clear();
              nHBFprocessed++;
              nEventsProcessed += nevents;
              auto found = nEvnetsHBF.find(nevents);
              if (found == nEvnetsHBF.end()) {
                nEvnetsHBF[nevents] = 1;
              } else {
                found->second++;
              }
            } else {
              LOG(debug) << "New HBF or Timeframe";
              hbfbuffer.clear();
              currentir.bc = o2::raw::RDHUtils::getTriggerBC(rdh);
              currentir.orbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);
            }
          } else {
            LOG(error) << "Found unknown trigger" << std::bitset<32>(trigger);
          }
          currentpos += o2::raw::RDHUtils::getOffsetToNext(rdh);
          continue;
        }
        if (o2::raw::RDHUtils::getStop(rdh)) {
          LOG(error) << "Unexpected stop";
        }

        // non-0 payload size:
        auto payloadsize = o2::raw::RDHUtils::getMemorySize(rdh) - o2::raw::RDHUtils::getHeaderSize(rdh);
        int endpoint = static_cast<int>(o2::raw::RDHUtils::getEndPointID(rdh));
        LOG(debug) << "Next RDH: ";
        LOG(debug) << "Found endpoint              " << endpoint;
        LOG(debug) << "Found trigger BC:           " << o2::raw::RDHUtils::getTriggerBC(rdh);
        LOG(debug) << "Found trigger Oribt:        " << o2::raw::RDHUtils::getTriggerOrbit(rdh);
        LOG(debug) << "Found payload size:         " << payloadsize;
        LOG(debug) << "Found offset to next:       " << o2::raw::RDHUtils::getOffsetToNext(rdh);
        LOG(debug) << "Stop bit:                   " << (o2::raw::RDHUtils::getStop(rdh) ? "yes" : "no");
        LOG(debug) << "Number of GBT words:        " << (payloadsize * sizeof(char) / sizeof(o2::focal::PadGBTWord));
        auto page_payload = dataBuffer.subspan(currentpos + o2::raw::RDHUtils::getHeaderSize(rdh), payloadsize);
        std::copy(page_payload.begin(), page_payload.end(), std::back_inserter(hbfbuffer));
        currentpos += o2::raw::RDHUtils::getOffsetToNext(rdh);
      }
    }
    reader.setNextTFToRead(++tfID);
    nTFprocessed++;
  }
  rootfilewriter->Write();
  LOG(info) << "Processed " << nTFprocessed << " timeframes, " << nHBFprocessed << " HBFs";
  LOG(info) << "Analyzed " << nEventsProcessed << " events:";
  LOG(info) << "=============================================================";
  for (auto& [nevents, nHBF] : nEvnetsHBF) {
    LOG(info) << "  " << nevents << " event(s)/HBF: " << nHBF << " HBFs ...";
  }
}