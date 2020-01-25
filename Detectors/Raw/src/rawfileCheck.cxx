// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   rawfileCheck.h
/// @author ruben.shahoyan@cern.ch
/// @brief  Checker for raw data conformity with CRU format

#include "DetectorsRaw/RawFileReader.h"
#include <TStopwatch.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace bpo = boost::program_options;

int main(int argc, char* argv[])
{
  o2::raw::RawFileReader reader;
  std::vector<std::string> fnames;
  bpo::variables_map vm;
  bpo::options_description descOpt("Options");
  descOpt.add_options()("help,h", "print this help message.")("verbosity,v", bpo::value<int>()->default_value(reader.getVerbosity()), "1: print RDH on error, 2: print all RDH")("spsize,s", bpo::value<int>()->default_value(reader.getNominalSPageSize()), "nominal super-page size in bytes")("hbfpertf,t", bpo::value<int>()->default_value(reader.getNominalHBFperTF()), "nominal number of HBFs per TF");

  bpo::options_description hiddenOpt("hidden");
  hiddenOpt.add_options()("input", bpo::value(&fnames)->composing(), "");

  bpo::options_description fullOpt("cmd");
  fullOpt.add(descOpt).add(hiddenOpt);

  bpo::positional_options_description posOpt;
  posOpt.add("input", -1);
  try {
    bpo::store(bpo::command_line_parser(argc, argv)
                 .options(fullOpt)
                 .positional(posOpt)
                 .allow_unregistered()
                 .run(),
               vm);
    bpo::notify(vm);
    if (argc == 1 || vm.count("help") || fnames.empty()) {
      std::cout << "Usage:   " << argv[0] << " [options] file0 [... fileN]" << std::endl;
      std::cout << descOpt << std::endl;
      return 0;
    }

  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing command line arguments\n";
    std::cerr << "Usage:   " << argv[0] << " [options] file0 [... fileN]" << std::endl;
    std::cerr << descOpt << std::endl;
    return -1;
  }

  reader.setVerbosity(vm["verbosity"].as<int>());
  reader.setNominalSPageSize(vm["spsize"].as<int>());
  reader.setNominalHBFperTF(vm["hbfpertf"].as<int>());

  for (int i = 0; i < fnames.size(); i++) {
    reader.addFile(fnames[i]);
  }

  TStopwatch sw;
  sw.Start();
  reader.setCheckErrors(true);

  o2::raw::RawFileReader::RDH rdh;
  LOG(INFO) << "RawDataHeader v" << int(rdh.version) << " is assumed";
  reader.init();

  sw.Print();

  return 0;
}
