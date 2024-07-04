// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test EMCAL Reconstruction
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <set>
#include <TRandom.h>
#include <EMCALBase/TriggerMappingV2.h>
#include <EMCALReconstruction/TRUDataHandler.h>

namespace o2
{
namespace emcal
{

BOOST_AUTO_TEST_CASE(TRUDataHandler_test)
{
  o2::emcal::TRUDataHandler testhandler;

  // no patch set
  BOOST_CHECK_EQUAL(testhandler.hasAnyPatch(), false);
  for (int ipatch = 0; ipatch < o2::emcal::TriggerMappingV2::PATCHESINTRU; ipatch++) {
    BOOST_CHECK_EQUAL(testhandler.hasPatch(ipatch), false);
  }

  // set all patches with L0 time 8
  for (int ipatch = 0; ipatch < o2::emcal::TriggerMappingV2::PATCHESINTRU; ipatch++) {
    testhandler.setPatch(ipatch, 8);
  }
  BOOST_CHECK_EQUAL(testhandler.hasAnyPatch(), true);
  for (int ipatch = 0; ipatch < o2::emcal::TriggerMappingV2::PATCHESINTRU; ipatch++) {
    BOOST_CHECK_EQUAL(testhandler.hasPatch(ipatch), true);
    BOOST_CHECK_EQUAL(testhandler.getPatchTime(ipatch), 8);
  }

  // test reset
  testhandler.reset();
  BOOST_CHECK_EQUAL(testhandler.hasAnyPatch(), false);
  for (int ipatch = 0; ipatch < o2::emcal::TriggerMappingV2::PATCHESINTRU; ipatch++) {
    BOOST_CHECK_EQUAL(testhandler.hasPatch(ipatch), false);
  }

  // test error handling
  for (int8_t index = o2::emcal::TriggerMappingV2::PATCHESINTRU; index < CHAR_MAX; index++) {
    BOOST_CHECK_EXCEPTION(testhandler.hasPatch(index), o2::emcal::TRUDataHandler::PatchIndexException, [index](const o2::emcal::TRUDataHandler::PatchIndexException& e) { return e.getIndex() == index; });
    BOOST_CHECK_EXCEPTION(testhandler.setPatch(index, 8), o2::emcal::TRUDataHandler::PatchIndexException, [index](const o2::emcal::TRUDataHandler::PatchIndexException& e) { return e.getIndex() == index; });
  }

  for (int iiter = 0; iiter < 100; iiter++) {
    // For 100 iterations simulate patch index and time
    std::map<uint8_t, uint8_t> patchtimes;
    int npatches_expect = static_cast<int>(gRandom->Uniform(0, o2::emcal::TriggerMappingV2::PATCHESINTRU));
    while (patchtimes.size() < npatches_expect) {
      auto patchindex = static_cast<int>(gRandom->Uniform(0, o2::emcal::TriggerMappingV2::PATCHESINTRU));
      if (patchtimes.find(patchindex) == patchtimes.end()) {
        auto patchtime = static_cast<int>(gRandom->Gaus(8, 1));
        if (patchtime >= 12) {
          patchtime = 11;
        }
        patchtimes[patchindex] = patchtime;
      }
    }
    o2::emcal::TRUDataHandler iterhandler;
    iterhandler.setFired(npatches_expect > 0);
    for (auto [patchindex, patchtime] : patchtimes) {
      iterhandler.setPatch(patchindex, patchtime);
    }

    BOOST_CHECK_EQUAL(iterhandler.isFired(), npatches_expect > 0);
    BOOST_CHECK_EQUAL(iterhandler.hasAnyPatch(), npatches_expect > 0);
    for (auto ipatch = 0; ipatch < o2::emcal::TriggerMappingV2::PATCHESINTRU; ipatch++) {
      auto hasPatch = patchtimes.find(ipatch) != patchtimes.end();
      BOOST_CHECK_EQUAL(iterhandler.hasPatch(ipatch), hasPatch);
      if (hasPatch) {
        BOOST_CHECK_EQUAL(iterhandler.getPatchTime(ipatch), patchtimes[ipatch]);
      }
    }
  }
}

} // namespace emcal

} // namespace o2