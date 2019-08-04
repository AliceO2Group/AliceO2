// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFlow/TimeframeParser.h"
#include "fairmq/FairMQParts.h"
#include <vector>
#include <string>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <fstream>

// A simple tool which verifies timeframe files
int main(int argc, char** argv)
{
  int c;
  opterr = 0;

  while ((c = getopt(argc, argv, "")) != -1) {
    switch (c) {
      case '?':
        if (isprint(optopt))
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf(stderr,
                  "Unknown option character `\\x%x'.\n",
                  optopt);
        return 1;
      default:
        abort();
    }
  }

  std::vector<std::string> filenames;
  for (size_t index = optind; index < argc; index++) {
    filenames.emplace_back(std::string(argv[index]));
  }

  for (auto&& fn : filenames) {
    LOG(INFO) << "Processing file" << fn << "\n";
    std::ifstream s(fn);
    FairMQParts parts;
    auto onAddParts = [](FairMQParts& p, char* buffer, size_t size) {
    };
    auto onSend = [](FairMQParts& p) {
    };

    try {
      o2::data_flow::streamTimeframe(s, onAddParts, onSend);
    } catch (std::runtime_error& e) {
      LOG(ERROR) << e.what() << std::endl;
      exit(1);
    }
  }
}
