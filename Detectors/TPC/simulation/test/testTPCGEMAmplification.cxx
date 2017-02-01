#define BOOST_TEST_MODULE Test TPC GEMAmplification
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCSimulation/GEMAmplification.h"
#include "TPCSimulation/Constants.h"

#include "TH1D.h"
#include "TF1.h"

namespace AliceO2 {
namespace TPC {

  BOOST_AUTO_TEST_CASE(GEMamplification_test)
  {
    GEMAmplification gemStack(EFFGAINGEM1, EFFGAINGEM2, EFFGAINGEM3, EFFGAINGEM4);
    TH1D hTest("hTest", "", 10000, 0, 1000000);
    TF1 gaus("gaus", "gaus");  
    
    for(int i=0; i < 100000; ++i) {
      hTest.Fill(gemStack.getStackAmplification(155));
    }
    
    hTest.Fit("gaus", "Q0");
    float energyResolution = gaus.GetParameter(2)/gaus.GetParameter(1);
    bool testEnergyResolution = (energyResolution < 0.20);                      /// @todo should be decreased to 14 %
    BOOST_CHECK(testEnergyResolution);
    BOOST_CHECK_CLOSE(gaus.GetParameter(1), (155.f * EFFGAINGEM1 * EFFGAINGEM2 * EFFGAINGEM3 * EFFGAINGEM4), 25.f);  /// @todo should be more restrictive
  }
}
} 