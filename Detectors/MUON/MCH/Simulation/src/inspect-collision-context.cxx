// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include "Framework/Logger.h"
#include <TFile.h>
#include "SimulationDataFormat/DigitizationContext.h"
#include <gsl/span>
#include "CommonDataFormat/InteractionRecord.h"
#include <limits>

namespace po = boost::program_options;

void report(gsl::span<o2::InteractionTimeRecord> irs, int threshold, bool verbose)
{
  o2::InteractionTimeRecord ir0(std::numeric_limits<double>::max());
  int tooClose{0};
  for (auto ir : irs) {
    if (verbose) {
      std::cout << ir;
    }
    auto d = ir.differenceInBC(ir0);
    if (d >= 0 && d < threshold) {
      if (verbose) {
        std::cout << " **** BC distance to previous " << d;
      }
      ++tooClose;
    }
    if (verbose) {
      std::cout << "\n";
    }
    ir0 = ir;
  }
  if (verbose) {
    std::cout << "Number of IR(s) strictly below the " << threshold << " BC distance limit : "
              << tooClose << "\n";
  } else {
    std::cout << tooClose << "\n";
  }
}

int main(int argc, char* argv[])
{
  po::options_description generic("options");
  po::variables_map vm;

  // clang-format off
  generic.add_options()
      ("help,h", "produce help message")
      ("input-file,i",po::value<std::string>()->default_value("collisioncontext.root"),"input file name")
      ("min-distance,d",po::value<int>()->default_value(4), "min distance between IRs to consider as a problem")
      ("verbose,v",po::value<bool>()->default_value(false),"verbose output");
  // clang-format on

  po::options_description cmdline;
  cmdline.add(generic);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << generic << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  po::notify(vm);

  // first things first : check the input path actually exists
  std::string input = vm["input-file"].as<std::string>();

  TFile fin(input.c_str());
  if (!fin.IsOpen()) {
    LOGP(fatal, "could not open input file {}", input);
    return -1;
  }
  auto context = fin.Get<o2::steer::DigitizationContext>("DigitizationContext");
  if (!context) {
    std::cout << "Could not get context\n";
    return 1;
  }
  report(context->getEventRecords(), vm["min-distance"].as<int>(), vm["verbose"].as<bool>());
  return 0;
}
