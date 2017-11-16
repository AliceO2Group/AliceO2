// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTPCElectronTransport.cxx
/// \brief This task tests the ElectronTransport module of the TPC digitization
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#define BOOST_TEST_MODULE Test TPC ElectronTransport
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCSimulation/ElectronTransport.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterDetector.h"

#include "TH1D.h"
#include "TF1.h"

namespace o2 {
namespace TPC {

  /// \brief Test 1 of the getElectronDrift function
  /// A defined position is given to the getElectronDrift function
  /// in which the position is randomly smeared according to a Gaussian
  /// We then compare the resulting mean and width to the expected one
  ///
  /// Precision: 0.5 %.
  BOOST_AUTO_TEST_CASE(ElectronDiffusion_test1)
  {
    const static ParameterGas &gasParam = ParameterGas::defaultInstance();
    const static ParameterDetector &detParam = ParameterDetector::defaultInstance();
    const GlobalPosition3D posEle(10.f, 10.f, 10.f);
    TH1D hTestDiffX("hTestDiffX", "", 500, posEle.X()-10., posEle.X()+10.);
    TH1D hTestDiffY("hTestDiffY", "", 500, posEle.Y()-10., posEle.Y()+10.);
    TH1D hTestDiffZ("hTestDiffZ", "", 500, posEle.Z()-10., posEle.Z()+10.);
    
    TF1 gausX("gausX", "gaus");
    TF1 gausY("gausY", "gaus");
    TF1 gausZ("gausZ", "gaus");
    
    static ElectronTransport electronTransport;
    
    for(int i=0; i<500000; ++i) {
      const GlobalPosition3D posEleDiff = electronTransport.getElectronDrift(posEle);
      hTestDiffX.Fill(posEleDiff.X());
      hTestDiffY.Fill(posEleDiff.Y());
      hTestDiffZ.Fill(posEleDiff.Z());
    }

    hTestDiffX.Fit("gausX", "Q0");
    hTestDiffY.Fit("gausY", "Q0");
    hTestDiffZ.Fit("gausZ", "Q0");
   
    // check whether the mean of the gaussian fit matches the starting point
    BOOST_CHECK_CLOSE(gausX.GetParameter(1), posEle.X(), 0.5);
    BOOST_CHECK_CLOSE(gausY.GetParameter(1), posEle.Y(), 0.5);
    BOOST_CHECK_CLOSE(gausZ.GetParameter(1), posEle.Z(), 0.5);
    
    // check whether the width of the distribution matches the expected one
    const float sigT = std::sqrt(detParam.getTPClength()-posEle.Z()) * gasParam.getDiffT();
    const float sigL = std::sqrt(detParam.getTPClength()-posEle.Z()) * gasParam.getDiffL();
        
    BOOST_CHECK_CLOSE(gausX.GetParameter(2), sigT, 0.5);
    BOOST_CHECK_CLOSE(gausY.GetParameter(2), sigT, 0.5);
    BOOST_CHECK_CLOSE(gausZ.GetParameter(2), sigL, 0.5);
  }

  /// \brief Test 2 of the getElectronDrift function
  /// We drift the electrons by one cm, then the width of
  /// the smeared distributions should be exactly the same
  /// as the diffusion coefficients
  ///
  /// Precision: 0.5 %.
  BOOST_AUTO_TEST_CASE(ElectronDiffusion_test2)
  {
    const static ParameterGas &gasParam = ParameterGas::defaultInstance();
    const static ParameterDetector &detParam = ParameterDetector::defaultInstance();
    const GlobalPosition3D posEle(1.f, 1.f, detParam.getTPClength()-1.f);
    TH1D hTestDiffX("hTestDiffX", "", 500, posEle.X()-1., posEle.X()+1.);
    TH1D hTestDiffY("hTestDiffY", "", 500, posEle.Y()-1., posEle.Y()+1.);
    TH1D hTestDiffZ("hTestDiffZ", "", 500, posEle.Z()-1., posEle.Z()+1.);
    
    TF1 gausX("gausX", "gaus");
    TF1 gausY("gausY", "gaus");
    TF1 gausZ("gausZ", "gaus");
    
    static ElectronTransport electronTransport;
    
    for(int i=0; i<500000; ++i) {
      const GlobalPosition3D posEleDiff = electronTransport.getElectronDrift(posEle);
      hTestDiffX.Fill(posEleDiff.X());
      hTestDiffY.Fill(posEleDiff.Y());
      hTestDiffZ.Fill(posEleDiff.Z());
    }
    
    hTestDiffX.Fit("gausX", "Q0");
    hTestDiffY.Fit("gausY", "Q0");
    hTestDiffZ.Fit("gausZ", "Q0");
   
    // check whether the mean of the gaussian fit matches the starting point
    BOOST_CHECK_CLOSE(gausX.GetParameter(1), posEle.X(), 0.5);
    BOOST_CHECK_CLOSE(gausY.GetParameter(1), posEle.Y(), 0.5);
    BOOST_CHECK_CLOSE(gausZ.GetParameter(1), posEle.Z(), 0.5);
    
    // check whether the width of the distribution matches the expected one
    BOOST_CHECK_CLOSE(gausX.GetParameter(2), gasParam.getDiffT(), 0.5);
    BOOST_CHECK_CLOSE(gausY.GetParameter(2), gasParam.getDiffT(), 0.5);
    BOOST_CHECK_CLOSE(gausZ.GetParameter(2), gasParam.getDiffL(), 0.5);
  }
  
  /// \brief Test of the isElectronAttachment function
  /// We let the electrons drift for 100 us and compare the fraction
  /// of lost electrons to the expected value
  ///
  /// Precision: 0.1 %.
  BOOST_AUTO_TEST_CASE(ElectronAttatchment_test_1)
  {
    const static ParameterGas &gasParam = ParameterGas::defaultInstance();
    static ElectronTransport electronTransport;

    const float driftTime = 100.f;
    float lostElectrons = 0;
    const float nEvents = 500000;
    for(int i=0; i<nEvents; ++i) {
      if(electronTransport.isElectronAttachment(driftTime)) {
        ++ lostElectrons;
      }
    }

    BOOST_CHECK_CLOSE(lostElectrons/nEvents, gasParam.getAttachmentCoefficient() * gasParam.getOxygenContent() * driftTime, 0.1);
  }
}
}
