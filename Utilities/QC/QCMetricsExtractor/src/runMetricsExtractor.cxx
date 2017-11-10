// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "QCMetricsExtractor/MetricsExtractor.h"

#include <iostream>

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

namespace bpo = boost::program_options;
using namespace o2::qc;

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::cerr << "Invalid number of program arguments: " << argc << std::endl;
  }

  const char* TEST_NAME = argv[1];

  bpo::options_description options("ui-custom-cmd options");
  options.add_options()("help,h", "Produce help message");

  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(options).run(), vm);
  bpo::notify(vm);

  MetricsExtractor MetricsExtractor(TEST_NAME);
  MetricsExtractor.runMetricsExtractor();
}
