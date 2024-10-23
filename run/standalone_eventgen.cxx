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

#include "SimulationDataFormat/MCTrack.h"
#include <Generators/GeneratorService.h>
#include <CommonUtils/ConfigurableParam.h>
#include <CommonUtils/RngHelper.h>
#include <TStopwatch.h> // simple timer from ROOT
#include <DetectorsCommonDataFormats/DetectorNameConf.h>
#include <DetectorsBase/Detector.h>
#include <TFile.h>
#include <memory>

// A simple, non-DPL, executable for event generation

struct GeneratorTask {
  // For readability to indicate where counting certain things (such as events or timeframes) should be of the same order of magnitude
  typedef uint64_t GenCount;
  std::string generator = "external"; //{"generator", "boxgen", "Name of generator"};
  GenCount eventNum = 3; // {"nEvents", 1, "Number of events"};
  std::string trigger = ""; //{"trigger", "", "Trigger type"}; //
  // std::string iniFile = "/home/swenzel/alisw/O2DPG/MC/config/ALICE3/ini/pythia8_pp_136tev.ini"; //{"configFile", "", "INI file containing configurable parameters"};
  std::string iniFile = "/home/swenzel/alisw/O2DPG/MC/config/PWGEM/ini/GeneratorEMCocktail.ini";
  std::string params = ""; // {"configKeyValues", "", "configurable params - configuring event generation internals"};
  long seed = 0;
  int aggregate = 10;
  std::string vtxModeArg = "kDiamondParam";
  int64_t ttl = -1; // "time-limit", -1, "Maximum run time limit in seconds (default no limit)"};
  std::string outputPrefix = "";// {"output", "", "Optional prefix for kinematics files written on disc. If non-empty, files <prefix>_Kine.root + <prefix>_MCHeader.root will be created."};
  GenCount nEvents = 0;
  GenCount eventCounter = 0;
  GenCount tfCounter = 0;
  std::unique_ptr<TFile> outfile{};
  std::unique_ptr<TTree> outtree{};

  // a pointer because object should only be constructed in the device (not during DPL workflow setup)
  std::unique_ptr<o2::eventgen::GeneratorService> genservice;
  TStopwatch timer;

  void init()
  {
    genservice.reset(new o2::eventgen::GeneratorService);
    o2::utils::RngHelper::setGRandomSeed(seed);
    nEvents = eventNum;
    // helper to parse vertex option; returns true if parsing ok, false if failure
    o2::conf::VertexMode vtxmode;
    if (!(o2::conf::SimConfig::parseVertexModeString(vtxModeArg, vtxmode))) {
      LOG(error) << "Could not parse vtxMode";
    }

    // update config key params
    o2::conf::ConfigurableParam::updateFromFile(iniFile);
    o2::conf::ConfigurableParam::updateFromString((std::string)params);
    // initialize the service
    if (vtxmode == o2::conf::VertexMode::kDiamondParam) {
      genservice->initService(generator, trigger, o2::eventgen::DiamondParamVertexOption());
    } else if (vtxmode == o2::conf::VertexMode::kNoVertex) {
      genservice->initService(generator, trigger, o2::eventgen::NoVertexOption());
    } else if (vtxmode == o2::conf::VertexMode::kCCDB) {
      LOG(warn) << "Not yet supported. This needs definition of a timestamp and fetching of the MeanVertex CCDB object";
    }
    timer.Start();

    if (outputPrefix.size() > 0 && !outfile.get()) {
      auto kineoutfilename = o2::base::NameConf::getMCKinematicsFileName(outputPrefix.c_str());
      outfile.reset(new TFile(kineoutfilename.c_str(), "RECREATE"));
      outtree.reset(new TTree("o2sim", "o2sim"));
    }
  }

  void run()
  {
    static int i = 0;
    LOG(warn) << "timeframe " << i;
    i++;
    std::vector<o2::MCTrack> mctracks;
    o2::dataformats::MCEventHeader mcheader;
    std::vector<o2::MCTrack> accum;
    std::vector<o2::dataformats::MCEventHeader> accumHeader;
    auto mctrack_ptr = &mctracks;
    if (outfile.get()) {
      auto br = o2::base::getOrMakeBranch(*outtree, "MCTrack", &mctrack_ptr);
      br->SetAddress(&mctrack_ptr);
    }

    auto toDoThisBatch = std::min((GenCount)aggregate, nEvents - eventCounter);
    LOG(info) << "Generating " << toDoThisBatch << " events";

    for (auto i = 0; i < toDoThisBatch; ++i) {
      mctracks.clear();
      genservice->generateEvent_MCTracks(mctracks, mcheader);
      // pc.outputs().snapshot(Output{"MC", "MCHEADER", 0}, mcheader);
      // pc.outputs().snapshot(Output{"MC", "MCTRACKS", 0}, mctracks);
      LOG(info) << "generated " << mctracks.size() << " tracks";
      std::copy(mctracks.begin(),mctracks.end(),std::back_inserter(accum));
      accumHeader.push_back(mcheader);
      ++eventCounter;

      // LOG(info) << "mctracks container cap " << mctracks.capacity() << " vs size " << mctracks.size();
      if (outfile.get() && outtree.get()) {
        outtree->Fill();
      }
    }

    LOG(info) << "Size of tracks accum " << accum.size() * sizeof(o2::MCTrack) << " bytes";

    // report number of TFs injected for the rate limiter to work
    ++tfCounter;
    bool time_expired = false;
    if (ttl > 0) {
      timer.Stop();
      time_expired = timer.RealTime() > ttl;
      timer.Start(false);
      if (time_expired) {
        LOG(info) << "TTL expired after " << eventCounter << " events ... sending end-of-stream";
      }
    }
    if (eventCounter >= nEvents || time_expired) {
      // write out data to disc if asked
      if (outfile.get()) {
        outtree->SetEntries(eventCounter);
        outtree->Write();
        outfile->Close();
      }
    }
  } // end run
}; // end struct

int main() {
  GeneratorTask task;
  task.init();
  for (int i=0; i < task.eventNum/task.aggregate + 1; ++i) {
    task.run();
  }
}