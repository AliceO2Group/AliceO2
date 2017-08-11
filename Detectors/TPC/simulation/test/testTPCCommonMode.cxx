// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTPCCommonMode.cxx
/// \brief This task tests the CommonModeContainer of the TPC digitization
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#define BOOST_TEST_MODULE Test TPC CommonModeContainer
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCSimulation/CommonModeContainer.h"
#include "TPCBase/Mapper.h"

namespace o2 {
namespace TPC {

  /// \brief Test of the CommonModeContainer
  /// A couple of values are filled into a CommonModeContainer and we check whether we get the expected results
  BOOST_AUTO_TEST_CASE(CommonMode_test1)
  {
    static const Mapper& mapper = Mapper::instance();
    CommonModeContainer c;

    /// make sure the container is empty
    for(int iCRU = 10; iCRU<CRU::MaxCRU; ++iCRU) {
      for(int iTime = 0; iTime < static_cast<int>(c.getNtimeBins()); ++ iTime) {
        BOOST_CHECK_CLOSE(c.getCommonMode(iCRU, iTime), 0.f, 1E-12);
      }
    }

    const float nPadsIROC  = static_cast<float>(mapper.getPadsInIROC());
    const float nPadsOROC1 = static_cast<float>(mapper.getPadsInOROC1());
    const float nPadsOROC2 = static_cast<float>(mapper.getPadsInOROC2());
    const float nPadsOROC3 = static_cast<float>(mapper.getPadsInOROC3());

    /// IROC, CRU 0 - 3
    c.addDigit(0,0,10);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,0), 10.f/nPadsIROC, 1E-12);
    c.addDigit(1,0,15);
    BOOST_CHECK_CLOSE(c.getCommonMode(1,0), 25.f/nPadsIROC, 1E-12);
    c.addDigit(2,0,20);
    BOOST_CHECK_CLOSE(c.getCommonMode(2,0), 45.f/nPadsIROC, 1E-12);
    c.addDigit(3,0,25);
    BOOST_CHECK_CLOSE(c.getCommonMode(3,0), 70.f/nPadsIROC, 1E-12);
    c.addDigit(3,1,5);
    BOOST_CHECK_CLOSE(c.getCommonMode(3,0), 70.f/nPadsIROC, 1E-12);

    /// OROC 1, CRU 4 - 5
    c.addDigit(4,0,5);
    BOOST_CHECK_CLOSE(c.getCommonMode(4,0), 5.f/nPadsOROC1, 1E-12);
    c.addDigit(5,0,10);
    BOOST_CHECK_CLOSE(c.getCommonMode(5,0), 15.f/nPadsOROC1, 1E-12);
    c.addDigit(4,1,6);
    BOOST_CHECK_CLOSE(c.getCommonMode(5,0), 15.f/nPadsOROC1, 1E-12);

    /// OROC 2, CRU 6 - 7
    c.addDigit(6,0,35);
    BOOST_CHECK_CLOSE(c.getCommonMode(6,0), 35.f/nPadsOROC2, 1E-12);
    c.addDigit(7,0,10);
    BOOST_CHECK_CLOSE(c.getCommonMode(7,0), 45.f/nPadsOROC2, 1E-12);
    c.addDigit(6,1,7);
    BOOST_CHECK_CLOSE(c.getCommonMode(7,0), 45.f/nPadsOROC2, 1E-12);

    /// OROC 3, CRU 7 - 8
    c.addDigit(8,0,10);
    BOOST_CHECK_CLOSE(c.getCommonMode(8,0), 10.f/nPadsOROC3, 1E-12);
    c.addDigit(9,0,10);
    BOOST_CHECK_CLOSE(c.getCommonMode(9,0), 20.f/nPadsOROC3, 1E-12);
    c.addDigit(9,1,8);
    BOOST_CHECK_CLOSE(c.getCommonMode(9,0), 20.f/nPadsOROC3, 1E-12);

    /// Check that there is no cross-talk
    BOOST_CHECK_CLOSE(c.getCommonMode(0,1), 5.f/nPadsIROC, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(1,1), 5.f/nPadsIROC, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(2,1), 5.f/nPadsIROC, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(3,1), 5.f/nPadsIROC, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(4,1), 6.f/nPadsOROC1, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(5,1), 6.f/nPadsOROC1, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(6,1), 7.f/nPadsOROC2, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(7,1), 7.f/nPadsOROC2, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(8,1), 8.f/nPadsOROC3, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(9,1), 8.f/nPadsOROC3, 1E-12);

    /// now make sure the rest is not used
    for(int iCRU = 10; iCRU<CRU::MaxCRU; ++iCRU) {
      BOOST_CHECK_CLOSE(c.getCommonMode(iCRU,1), 0.f, 1E-12);
    }
  }

  /// \brief Test of the CommonModeContainer in continuous mode
  /// A couple of values are filled into a CommonModeContainer and we check whether we get the expected results
  BOOST_AUTO_TEST_CASE(CommonMode_test2)
  {
    static const Mapper& mapper = Mapper::instance();
    CommonModeContainer c;
    c.addDigit(0,1,1);
    c.addDigit(0,10,10);
    c.addDigit(0,15,15);
    c.addDigit(0,20,20);
    /// We checked above that the container is thoroughly filled

    /// In triggered mode all values should be zero now
    c.cleanUp(5, false);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,1), 0.f, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,10), 0.f, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,15), 0.f, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,20), 0.f, 1E-12);

    /// Fill the container again
    c.addDigit(0,1,1);
    c.addDigit(0,10,10);
    c.addDigit(0,15,15);
    c.addDigit(0,20,20);

    const float nPadsIROC  = static_cast<float>(mapper.getPadsInIROC());

    /// In continuous mode only those values with time bins smaller than 5 should be invalidated
    c.cleanUp(5);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,1), 9999.f, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,10), 10.f/nPadsIROC, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,15), 15.f/nPadsIROC, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,20), 20.f/nPadsIROC, 1E-12);
    /// In continuous mode only those values with time bins smaller than 12 should be invalidated
    c.cleanUp(12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,1), 9999.f, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,10), 9999.f, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,15), 15.f/nPadsIROC, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,20), 20.f/nPadsIROC, 1E-12);
    /// In continuous mode only those values with time bins smaller than 18 should be invalidated
    c.cleanUp(18);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,1), 9999.f, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,10), 9999.f, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,15), 9999.f, 1E-12);
    BOOST_CHECK_CLOSE(c.getCommonMode(0,20), 20.f/nPadsIROC, 1E-12);
  }
}
}
