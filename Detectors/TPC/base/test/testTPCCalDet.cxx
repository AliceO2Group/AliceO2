// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TPC CalDet class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/range/combine.hpp>
#include <boost/test/unit_test.hpp>
#include <vector>
#include <limits>

#include "TMath.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/CalDet.h"
#include "TFile.h"

namespace o2
{
namespace tpc
{

//templated euqality check
// for integer one would need a specialisation to check for == instead of <
template <typename T>
bool isEqualAbs(T val1, T val2)
{
  return std::abs(val1 - val2) < std::numeric_limits<T>::epsilon();
}

BOOST_AUTO_TEST_CASE(CalArray_ROOTIO)
{
  //CalROC roc(PadSubset::ROC, 10);
  CalArray<unsigned> roc(PadSubset::ROC, 10);

  int iter = 0;
  //unsigned iter=0;
  for (auto& val : roc.getData()) {
    val = iter++;
  }

  auto f = TFile::Open("CalArray_ROOTIO.root", "recreate");
  f->WriteObject(&roc, "roc");
  delete f;

  //CalROC *rocRead = nullptr;
  CalArray<unsigned>* rocRead = nullptr;
  f = TFile::Open("CalArray_ROOTIO.root");
  f->GetObject("roc", rocRead);
  delete f;

  BOOST_REQUIRE(rocRead != nullptr);

  float sumROC = 0;
  for (auto const& val : boost::combine(roc.getData(), rocRead->getData())) {
    sumROC += (val.get<0>() - val.get<1>());
  }

  BOOST_CHECK_CLOSE(sumROC, 0., 1.E-12);
}

BOOST_AUTO_TEST_CASE(CalDet_ROOTIO)
{

  auto& mapper = Mapper::instance();
  const auto numberOfPads = mapper.getPadsInSector() * 36;

  CalPad padROC(PadSubset::ROC);
  CalPad padPartition(PadSubset::Partition);
  CalPad padRegion(PadSubset::Region);

  // ===| Fill Data |===========================================================
  int iter = 0;
  // --- ROC type
  padROC.setName("ROCData");
  for (auto& calArray : padROC.getData()) {
    for (auto& value : calArray.getData()) {
      value = iter++;
    }
  }

  // --- Partition type
  padPartition.setName("PartitionData");
  for (auto& calArray : padPartition.getData()) {
    for (auto& value : calArray.getData()) {
      value = iter++;
    }
  }

  // --- Region type
  padRegion.setName("RegionData");
  for (auto& calArray : padRegion.getData()) {
    for (auto& value : calArray.getData()) {
      value = iter++;
    }
  }

  // ===| dump all objects to file |============================================
  auto f = TFile::Open("CalDet.root", "recreate");
  f->WriteObject(&padROC, "CalDetROC");
  f->WriteObject(&padPartition, "CalDetPartition");
  f->WriteObject(&padRegion, "CalDetRegion");
  f->Close();
  delete f;

  // ===| read back all values |================================================
  CalPad* padROCRead = nullptr;
  CalPad* padPartitionRead = nullptr;
  CalPad* padRegionRead = nullptr;

  f = TFile::Open("CalDet.root");
  f->GetObject("CalDetROC", padROCRead);
  f->GetObject("CalDetPartition", padPartitionRead);
  f->GetObject("CalDetRegion", padRegionRead);

  delete f;

  BOOST_REQUIRE(padROCRead != nullptr);
  BOOST_REQUIRE(padPartitionRead != nullptr);
  BOOST_REQUIRE(padRegionRead != nullptr);

  // ===| compare values before and after |=====================================
  float sumROC = 0.f;
  float sumPartition = 0.f;
  float sumRegion = 0.f;

  int numberOfPadsROC = 0;
  int numberOfPadsPartition = 0;
  int numberOfPadsRegion = 0;

  for (auto const& arrays : boost::combine(padROC.getData(), padROCRead->getData())) {
    for (auto const& val : boost::combine(arrays.get<0>().getData(), arrays.get<1>().getData())) {
      sumROC += (val.get<0>() - val.get<1>());
      ++numberOfPadsROC;
    }
  }

  for (auto const& arrays : boost::combine(padPartition.getData(), padPartitionRead->getData())) {
    for (auto const& val : boost::combine(arrays.get<0>().getData(), arrays.get<1>().getData())) {
      sumPartition += (val.get<0>() - val.get<1>());
      ++numberOfPadsPartition;
    }
  }

  for (auto const& arrays : boost::combine(padRegion.getData(), padRegionRead->getData())) {
    for (auto const& val : boost::combine(arrays.get<0>().getData(), arrays.get<1>().getData())) {
      sumRegion += (val.get<0>() - val.get<1>());
      ++numberOfPadsRegion;
    }
  }

  // ===| checks |==============================================================
  BOOST_CHECK_EQUAL(padROC.getName(), padROCRead->getName());
  BOOST_CHECK_CLOSE(sumROC, 0.f, 1.E-12);
  BOOST_CHECK_EQUAL(numberOfPadsROC, numberOfPads);

  BOOST_CHECK_EQUAL(padPartition.getName(), padPartitionRead->getName());
  BOOST_CHECK_CLOSE(sumPartition, 0.f, 1.E-12);
  BOOST_CHECK_EQUAL(numberOfPadsPartition, numberOfPads);

  BOOST_CHECK_EQUAL(padRegion.getName(), padRegionRead->getName());
  BOOST_CHECK_CLOSE(sumRegion, 0.f, 1.E-12);
  BOOST_CHECK_EQUAL(numberOfPadsRegion, numberOfPads);
} // BOOST_AUTO_TEST_CASE

BOOST_AUTO_TEST_CASE(CalDet_Arithmetics)
{
  // data
  CalPad pad(PadSubset::ROC);

  // data 2 for testing operators on objects
  CalPad pad2(PadSubset::ROC);

  // for applying the operators on
  CalPad padCmp(PadSubset::ROC);

  // ===| fill with data |======================================================
  int iter = 0;
  // --- ROC type
  for (auto& calArray : pad.getData()) {
    for (auto& value : calArray.getData()) {
      value = iter++;
    }
  }

  iter = 1;
  for (auto& calArray : pad2.getData()) {
    for (auto& value : calArray.getData()) {
      value = iter++;
    }
  }

  //
  // ===| test operators with simple numbers |==================================
  //
  const float number = 0.2;
  bool isEqual = true;

  // + operator
  isEqual = true;
  padCmp = pad;
  padCmp += number;

  for (auto const& arrays : boost::combine(padCmp.getData(), pad.getData())) {
    for (auto const& val : boost::combine(arrays.get<0>().getData(), arrays.get<1>().getData())) {
      isEqual &= isEqualAbs(val.get<0>(), val.get<1>() + number);
    }
  }
  BOOST_CHECK_EQUAL(isEqual, true);

  // - operator
  isEqual = true;
  padCmp = pad;
  padCmp -= number;

  for (auto const& arrays : boost::combine(padCmp.getData(), pad.getData())) {
    for (auto const& val : boost::combine(arrays.get<0>().getData(), arrays.get<1>().getData())) {
      isEqual &= isEqualAbs(val.get<0>(), val.get<1>() - number);
    }
  }
  BOOST_CHECK_EQUAL(isEqual, true);

  // * operator
  isEqual = true;
  padCmp = pad;
  padCmp *= number;

  for (auto const& arrays : boost::combine(padCmp.getData(), pad.getData())) {
    for (auto const& val : boost::combine(arrays.get<0>().getData(), arrays.get<1>().getData())) {
      isEqual &= isEqualAbs(val.get<0>(), val.get<1>() * number);
    }
  }
  BOOST_CHECK_EQUAL(isEqual, true);

  // / operator
  isEqual = true;
  padCmp = pad;
  padCmp /= number;

  for (auto const& arrays : boost::combine(padCmp.getData(), pad.getData())) {
    for (auto const& val : boost::combine(arrays.get<0>().getData(), arrays.get<1>().getData())) {
      isEqual &= isEqualAbs(val.get<0>(), val.get<1>() / number);
    }
  }
  BOOST_CHECK_EQUAL(isEqual, true);

  //
  // ===| test operators with full object |=====================================
  //
  // + operator
  isEqual = true;
  padCmp = pad;
  padCmp += pad2;

  for (auto itpad = pad.getData().begin(), itpad2 = pad2.getData().begin(), itpadCmp = padCmp.getData().begin(); itpad != pad.getData().end(); ++itpad, ++itpad2, ++itpadCmp) {
    for (auto itval1 = (*itpad).getData().begin(), itval2 = (*itpad2).getData().begin(), itval3 = (*itpadCmp).getData().begin(); itval1 != (*itpad).getData().end(); ++itval1, ++itval2, ++itval3) {
      isEqual &= isEqualAbs(*itval3, *itval1 + *itval2);
    }
  }
  BOOST_CHECK_EQUAL(isEqual, true);

  // - operator
  isEqual = true;
  padCmp = pad;
  padCmp -= pad2;

  for (auto itpad = pad.getData().begin(), itpad2 = pad2.getData().begin(), itpadCmp = padCmp.getData().begin(); itpad != pad.getData().end(); ++itpad, ++itpad2, ++itpadCmp) {
    for (auto itval1 = (*itpad).getData().begin(), itval2 = (*itpad2).getData().begin(), itval3 = (*itpadCmp).getData().begin(); itval1 != (*itpad).getData().end(); ++itval1, ++itval2, ++itval3) {
      isEqual &= isEqualAbs(*itval3, *itval1 - *itval2);
    }
  }
  BOOST_CHECK_EQUAL(isEqual, true);

  // * operator
  isEqual = true;
  padCmp = pad;
  padCmp *= pad2;

  for (auto itpad = pad.getData().begin(), itpad2 = pad2.getData().begin(), itpadCmp = padCmp.getData().begin(); itpad != pad.getData().end(); ++itpad, ++itpad2, ++itpadCmp) {
    for (auto itval1 = (*itpad).getData().begin(), itval2 = (*itpad2).getData().begin(), itval3 = (*itpadCmp).getData().begin(); itval1 != (*itpad).getData().end(); ++itval1, ++itval2, ++itval3) {
      isEqual &= isEqualAbs(*itval3, *itval1 * *itval2);
    }
  }
  BOOST_CHECK_EQUAL(isEqual, true);

  // / operator
  isEqual = true;
  padCmp = pad;
  padCmp /= pad2;

  for (auto itpad = pad.getData().begin(), itpad2 = pad2.getData().begin(), itpadCmp = padCmp.getData().begin(); itpad != pad.getData().end(); ++itpad, ++itpad2, ++itpadCmp) {
    for (auto itval1 = (*itpad).getData().begin(), itval2 = (*itpad2).getData().begin(), itval3 = (*itpadCmp).getData().begin(); itval1 != (*itpad).getData().end(); ++itval1, ++itval2, ++itval3) {
      isEqual &= isEqualAbs(*itval3, *itval1 / *itval2);
    }
  }
  BOOST_CHECK_EQUAL(isEqual, true);
}

} // namespace tpc
} // namespace o2
