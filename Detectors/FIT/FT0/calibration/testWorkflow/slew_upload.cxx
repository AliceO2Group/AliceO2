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

/// \file slew_upload.cxx
/// \author Alla.Maevskaya@cern.ch

#include <boost/program_options.hpp>
#include <filesystem>
#include <TFile.h>
#include <TStopwatch.h>
#include "Framework/Logger.h"
#include <string>
#include <iomanip>
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "FT0Calibration/FT0CalibTimeSlewing.h"
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <string_view>

using o2::ccdb::BasicCCDBManager;
using o2::ccdb::CcdbApi;
using namespace o2::ft0;

namespace bpo = boost::program_options;

void slew_upload(const std::string& inFileName, const std::string& mergedFileName, int nFiles);

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       "Upload FT0 slewing corrections\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("input-file", bpo::value<std::string>()->default_value("collFT0.root"), "verbosity level");
    add_option("merged-file", bpo::value<std::string>()->default_value("FT0slewGraphs.root"), "input merged  file");
    add_option("number-of-files", bpo::value<int>()->default_value(1), "number of files to merge");
    add_option("configKeyValues", bpo::value<std::string>()->default_value(""), "comma-separated configKeyValues");

    opt_all.add(opt_general).add(opt_hidden);
    bpo::store(bpo::command_line_parser(argc, argv).options(opt_all).positional(opt_pos).run(), vm);

    if (vm.count("help")) {
      std::cout << opt_general << std::endl;
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
  o2::conf::ConfigurableParam::updateFromString(vm["configKeyValues"].as<std::string>());

  //  o2::conf::ConfigurableParam::updateFromString(vm["configKeyValues"].as<std::string>());
  slew_upload(vm["input-file"].as<std::string>(),
              vm["merged-file"].as<std::string>(),
              vm["number-of-files"].as<int>());

  return 0;
}

void slew_upload(const std::string& inFileName, const std::string& mergedFileName, int nFiles)
{
  TStopwatch swTot;
  swTot.Start();

  o2::ft0::FT0CalibTimeSlewing sl;
  sl.setSingleFileName(inFileName);
  sl.setMergedFileName(mergedFileName);
  sl.setNfiles(nFiles);
  sl.mergeFilesWithTree();
  for (int iCh = 0; iCh < o2::ft0 ::Geometry::Nchannels; ++iCh) {
    TH2F* hist = sl.getTimeAmpHist(iCh);
    if (hist->GetEntries() < 100000) {
      continue;
    }
    sl.fillGraph(iCh, hist);
  }
  std::array<TGraph, o2::ft0 ::Geometry::Nchannels> graphs = sl.getGraphs();
  TGraph& gr = graphs.at(29);
  gr.Print();
  CcdbApi api;
  std::map<std::string, std::string> metadata; // can be empty
  api.init(o2::base::NameConf::getCCDBServer()); // or http://localhost:8080 for a local installation
                                                 // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&graphs, "FT0/Calib/SlewingCorr", metadata);

  //
  swTot.Stop();
  swTot.Print();
}
