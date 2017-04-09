/// \file testTPCDigitContainer.cxx
/// \brief This task tests the DigitContainer of the TPC digitization
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#define BOOST_TEST_MODULE Test TPC DigitContainer
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TClonesArray.h"
#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/DigitMC.h"
#include "TPCSimulation/SAMPAProcessing.h"

namespace o2 {
namespace TPC {

  /// \brief Test of the DigitContainer
  /// A couple of values are filled into a DigitContainer and we check whether we get the same results after full conversion to digits
  BOOST_AUTO_TEST_CASE(DigitContainer_test1)
  {
    const SAMPAProcessing& sampa = SAMPAProcessing::instance();
    auto *mDigitContainer = new DigitContainer();

    std::vector<int> MCevent = {1, 250, 3, 62, 1000};
    std::vector<int> MCtrack = {22, 3, 4, 23, 523};
    std::vector<int> CRU     = {1, 23, 36, 53, 214};
    std::vector<int> Time    = {231, 2, 500, 230, 1};
    std::vector<int> Row     = {12, 5, 6, 2, 6};
    std::vector<int> Pad     = {1, 15, 14, 23, 5};
    std::vector<int> nEle    = {60, 100, 250, 1023, 2};

    for(int i=0; i<CRU.size(); ++i) {
      mDigitContainer->addDigit(MCevent[i], MCtrack[i], CRU[i], Time[i], Row[i], Pad[i], nEle[i]);
    }

    auto *mDigitsArray = new TClonesArray("o2::TPC::DigitMC");
    mDigitContainer->fillOutputContainer(mDigitsArray);

    BOOST_CHECK(CRU.size() == mDigitsArray->GetEntriesFast());

    int digits = 0;
    for(auto digitsObject : *mDigitsArray) {
      DigitMC *digit = static_cast<DigitMC *>(digitsObject);
      BOOST_CHECK(digit->getMCEvent(0) == MCevent[digits]);
      BOOST_CHECK(digit->getMCTrack(0) == MCtrack[digits]);
      BOOST_CHECK(digit->getCRU() == CRU[digits]);
      BOOST_CHECK(digit->getTimeStamp() == Time[digits]);
      BOOST_CHECK(digit->getRow() == Row[digits]);
      BOOST_CHECK(digit->getPad() == Pad[digits]);
      BOOST_CHECK(digit->getCharge() == static_cast<int>(sampa.getADCSaturation(nEle[digits])));
      ++digits;
    }
  }



  /// \brief Test of the DigitContainer
  /// A couple of values are into the very same voxel (CRU, TimeBin, Row, Pad) and we check whether the charges are properly summed up
  /// and that the MC labels are right
  BOOST_AUTO_TEST_CASE(DigitContainer_test2)
  {
    const SAMPAProcessing& sampa = SAMPAProcessing::instance();
    auto *mDigitContainer = new DigitContainer();

    std::vector<int> MCevent = { 1, 62,  1, 62, 62, 50, 62, 1, 1, 1};
    std::vector<int> MCtrack = {22,  3, 22,  3,  3, 70,  3, 7, 7, 7};
    std::vector<int> CRU     = {23, 23, 23, 23, 23, 23, 23, 23, 23, 23};
    std::vector<int> Time    = {231, 231, 231, 231, 231, 231, 231, 231, 231, 231};
    std::vector<int> Row     = {12, 12, 12, 12, 12, 12, 12, 12, 12, 12};
    std::vector<int> Pad     = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
    std::vector<int> nEle    = {60, 1, 252, 10, 2, 3, 5, 25, 24, 23};

    std::vector<int> MCeventSorted = {62, 1, 1, 50};
    std::vector<int> MCtrackSorted = {3, 7, 22, 70};

    int nEleSum = 0;
    for(int i=0; i<CRU.size(); ++i) {
      mDigitContainer->addDigit(MCevent[i], MCtrack[i], CRU[i], Time[i], Row[i], Pad[i], nEle[i]);
      nEleSum += nEle[i];
    }

    auto *mDigitsArray = new TClonesArray("o2::TPC::DigitMC");
    mDigitContainer->fillOutputContainer(mDigitsArray);

    BOOST_CHECK(mDigitsArray->GetEntriesFast() == 1);

    int digits = 0;
    for(auto digitsObject : *mDigitsArray) {
      DigitMC *digit = static_cast<DigitMC *>(digitsObject);
      for(int j=0; j<digit->getNumberOfMClabels(); ++j) {
        BOOST_CHECK(digit->getMCEvent(j) == MCeventSorted[j]);
        BOOST_CHECK(digit->getMCTrack(j) == MCtrackSorted[j]);
      }
      BOOST_CHECK(digit->getCRU() == CRU[digits]);
      BOOST_CHECK(digit->getTimeStamp() == Time[digits]);
      BOOST_CHECK(digit->getRow() == Row[digits]);
      BOOST_CHECK(digit->getPad() == Pad[digits]);
      BOOST_CHECK(digit->getCharge() == static_cast<int>(sampa.getADCSaturation(nEleSum)));
      ++digits;
    }
  }

}
}
