/// \file testTPCElectronTransport.cxx
/// \brief This task tests the ElectronTransport module of the TPC digitization
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#define BOOST_TEST_MODULE Test TPC ElectronTransport
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/Constants.h"

#include "TH1D.h"
#include "TF1.h"

namespace o2 {
namespace TPC {

  /// @brief Test 1 of the getElectronDrift function
  /// A defined position is given to the getElectronDrift function
  /// in which the position is randomly smeared according to a Gaussian
  /// We then compare the resulting mean and width to the expected one
  ///
  /// Precision: 0.5 %.
  BOOST_AUTO_TEST_CASE(ElectronDiffusion_test1)
  {
    const GlobalPosition3D posEle(10.f, 10.f, 250.f);
    TH1D hTestDiffX("hTestDiffX", "", 500, posEle.getX()-10., posEle.getX()+10.);
    TH1D hTestDiffY("hTestDiffY", "", 500, posEle.getY()-10., posEle.getY()+10.);
    TH1D hTestDiffZ("hTestDiffZ", "", 500, posEle.getZ()-10., posEle.getZ()+10.);
    
    TF1 gausX("gausX", "gaus");
    TF1 gausY("gausY", "gaus");
    TF1 gausZ("gausZ", "gaus");
    
    static ElectronTransport electronTransport;
    
    for(int i=0; i<500000; ++i) {
      const GlobalPosition3D posEleDiff = electronTransport.getElectronDrift(posEle);
      hTestDiffX.Fill(posEleDiff.getX());
      hTestDiffY.Fill(posEleDiff.getY());
      hTestDiffZ.Fill(posEleDiff.getZ());
    }

    hTestDiffX.Fit("gausX", "Q0");
    hTestDiffY.Fit("gausY", "Q0");
    hTestDiffZ.Fit("gausZ", "Q0");
   
    // check whether the mean of the gaussian fit matches the starting point
    BOOST_CHECK_CLOSE(gausX.GetParameter(1), posEle.getX(), 0.5);
    BOOST_CHECK_CLOSE(gausY.GetParameter(1), posEle.getY(), 0.5);
    BOOST_CHECK_CLOSE(gausZ.GetParameter(1), posEle.getZ(), 0.5);
    
    // check whether the width of the distribution matches the expected one
    float sigT = std::sqrt(posEle.getZ()) * DIFFT;
    float sigL = std::sqrt(posEle.getZ()) * DIFFL;
        
    BOOST_CHECK_CLOSE(gausX.GetParameter(2), sigT, 0.5);
    BOOST_CHECK_CLOSE(gausY.GetParameter(2), sigT, 0.5);
    BOOST_CHECK_CLOSE(gausZ.GetParameter(2), sigL, 0.5);
  }

  /// @brief Test 2 of the getElectronDrift function
  /// We drift the electrons by one cm, then the width of
  /// the smeared distributions should be exactly the same
  /// as the diffusion coefficients
  ///
  /// Precision: 0.5 %.
  BOOST_AUTO_TEST_CASE(ElectronDiffusion_test2)
  {
    const GlobalPosition3D posEle(1.f, 1.f, 1.f);
    TH1D hTestDiffX("hTestDiffX", "", 500, posEle.getX()-1., posEle.getX()+1.);
    TH1D hTestDiffY("hTestDiffY", "", 500, posEle.getY()-1., posEle.getY()+1.);
    TH1D hTestDiffZ("hTestDiffZ", "", 500, posEle.getZ()-1., posEle.getZ()+1.);
    
    TF1 gausX("gausX", "gaus");
    TF1 gausY("gausY", "gaus");
    TF1 gausZ("gausZ", "gaus");
    
    static ElectronTransport electronTransport;
    
    for(int i=0; i<500000; ++i) {
      const GlobalPosition3D posEleDiff = electronTransport.getElectronDrift(posEle);
      hTestDiffX.Fill(posEleDiff.getX());
      hTestDiffY.Fill(posEleDiff.getY());
      hTestDiffZ.Fill(posEleDiff.getZ());
    }
    
    hTestDiffX.Fit("gausX", "Q0");
    hTestDiffY.Fit("gausY", "Q0");
    hTestDiffZ.Fit("gausZ", "Q0");
   
    // check whether the mean of the gaussian fit matches the starting point
    BOOST_CHECK_CLOSE(gausX.GetParameter(1), posEle.getX(), 0.5);
    BOOST_CHECK_CLOSE(gausY.GetParameter(1), posEle.getY(), 0.5);
    BOOST_CHECK_CLOSE(gausZ.GetParameter(1), posEle.getZ(), 0.5);
    
    // check whether the width of the distribution matches the expected one
    BOOST_CHECK_CLOSE(gausX.GetParameter(2), DIFFT, 0.5);
    BOOST_CHECK_CLOSE(gausY.GetParameter(2), DIFFT, 0.5);
    BOOST_CHECK_CLOSE(gausZ.GetParameter(2), DIFFL, 0.5);
  }
  
  /// @brief Test of the isElectronAttachment function
  /// We let the electrons drift for 100 us and compare the fraction
  /// of lost electrons to the expected value
  ///
  /// Precision: 0.1 %.
  BOOST_AUTO_TEST_CASE(ElectronAttatchment_test_1)
  {
    static ElectronTransport electronTransport;

    float driftTime = 100.f;
    float lostElectrons = 0;
    float nEvents = 500000;
    for(int i=0; i<nEvents; ++i) {
      if(electronTransport.isElectronAttachment(driftTime)) {
        ++ lostElectrons;
      }
    }

    BOOST_CHECK_CLOSE(lostElectrons/nEvents, ATTCOEF * OXYCONT * driftTime, 0.1); 
  }
}
}