#include "DataFlow/TimeframeParser.h"
#include "fairmq/FairMQParts.h"
#include <vector>
#include <string>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

// A simple tool which verifies timeframe files
int main(int argc, char **argv) {
  int c;
  opterr = 0;

  while ((c = getopt (argc, argv, "")) != -1) {
    switch (c)
    {
    case '?':
      if (isprint (optopt))
        fprintf (stderr, "Unknown option `-%c'.\n", optopt);
      else
        fprintf (stderr,
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

  for (auto &&fn : filenames) {
    LOG(INFO) << "Processing file" << fn << "\n";
    std::ifstream s(fn);
    FairMQParts parts;
    auto onAddParts = [](FairMQParts &p, char *buffer, size_t size) {
    };
    auto onSend = [](FairMQParts &p) {
    };

    try {
      o2::DataFlow::streamTimeframe(s, onAddParts, onSend);
    } catch(std::runtime_error &e) {
      LOG(ERROR) << e.what() << std::endl;
      exit(1);
    }
  }
}
