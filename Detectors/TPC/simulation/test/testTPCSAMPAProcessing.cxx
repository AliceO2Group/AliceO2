#define BOOST_TEST_MODULE Test TPC SAMPAProcessing
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCSimulation/SAMPAProcessing.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "FairLogger.h"

namespace o2 {
namespace TPC {

  // read in the file on which the spline for the SAMPA saturation are based and compare the final spline to the contents of the file
  BOOST_AUTO_TEST_CASE(SAMPA_saturation_test)
  {
    const SAMPAProcessing& sampa = SAMPAProcessing::instance();

    std::string file = "SAMPA_saturation.dat";
    std::string inputDir;
    const char* aliceO2env = std::getenv("O2_ROOT");
    if (aliceO2env) {
      inputDir = aliceO2env;
    }
    inputDir += "/share/Detectors/TPC/files/";

    std::ifstream saturationFile(inputDir + file, std::ifstream::in);
    if (!saturationFile) {
      LOG(FATAL) << "TPC::SAMPAProcessing - Input file '" << inputDir + file << "' does not exist! No SAMPA saturation curve loaded!" << FairLogger::endl;
      BOOST_CHECK(false);
    }
    std::vector<std::pair<float, float>> saturation;
    for (std::string line; std::getline(saturationFile, line);) {
      float x, y;
      std::istringstream is(line);
      while (is >> x >> y) {
          saturation.emplace_back(x, y);
      }
    }

    for(int i=0; i<saturation.size(); ++i) {
      BOOST_CHECK(saturation[i].second == sampa.getADCSaturation(saturation[i].first));
    }
  }
}
}
