#define BOOST_TEST_MODULE Test TPC GEMAmplification
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCSimulation/GEMAmplification.h"
#include "TPCSimulation/Constants.h"

#include "TH1D.h"
#include "TF1.h"

namespace o2 {
namespace TPC {

  BOOST_AUTO_TEST_CASE(GEMamplification_test)
  {
    GEMAmplification gemStack;
    TH1D hTest("hTest", "", 10000, 0, 1000000);
    TF1 gaus("gaus", "gaus");  
    
    const int nEleIn = 158; /// Number of electrons liberated in Ne-CO2-N2 by an incident Fe-55 photon

    for(int i=0; i < 100000; ++i) {
      hTest.Fill(gemStack.getStackAmplification(nEleIn));
    }
    
    hTest.Fit("gaus", "Q0");
    float energyResolution = gaus.GetParameter(2)/gaus.GetParameter(1) *100.f;
    BOOST_CHECK_CLOSE(energyResolution, 12.1, 5);  /// we allow for 5% variation which is given by the uncertainty of the experimental determination of the energy resolution (12.1 +/- 0.5) %
    BOOST_CHECK_CLOSE(gaus.GetParameter(1)/static_cast<float>(nEleIn), (EFFGAINGEM1 * EFFGAINGEM2 * EFFGAINGEM3 * EFFGAINGEM4), 20.f);  /// @todo should be more restrictive
  }
}
} 
