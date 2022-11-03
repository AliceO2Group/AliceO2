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

#define BOOST_TEST_MODULE Test HBFUtils class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <algorithm>
#include <bitset>
#include <boost/test/unit_test.hpp>
#include "Steer/InteractionSampler.h"
#include "DetectorsRaw/HBFUtils.h"
#include "Headers/RAWDataHeader.h"
#include <TRandom.h>
#include <fairlogger/Logger.h>

// @brief test and demo for HBF sampling for simulated IRs
// @author ruben.shahoyan@cern.ch

namespace o2
{
BOOST_AUTO_TEST_CASE(HBFUtils)
{
  using RDH = o2::header::RAWDataHeaderV5;
  using IR = o2::InteractionRecord;

  const bool useContinuous = true;

  const auto& sampler = o2::raw::HBFUtils::Instance();

  // default sampler with BC filling like in TPC TDR
  o2::steer::InteractionSampler irSampler;
  irSampler.setInteractionRate(12000); // ~1.5 interactions per orbit
  irSampler.setFirstIR(sampler.getFirstIR());
  irSampler.init();

  int nIRs = 500;
  std::vector<o2::InteractionTimeRecord> irs(nIRs);
  irSampler.generateCollisionTimes(irs);

  LOG(info) << "Emulate RDHs for raw data between IRs " << irs.front() << " and " << irs.back();

  uint8_t packetCounter = 0;
  std::vector<o2::InteractionRecord> HBIRVec;
  auto irFrom = sampler.getFirstIR(); // TFs are counted from this IR
  int nHBF = 0, nHBFEmpty = 0, nTF = 0;
  int nHBFOpen = 0, nHBFClose = 0;
  RDH rdh;
  IR rdhIR;
  auto flushRDH = [&]() {
    bool empty = rdh.offsetToNext == sizeof(RDH);
    std::bitset<32> trig(rdh.triggerType);
    int hbfID = sampler.getHBF(rdhIR);
    auto tfhb = sampler.getTFandHBinTF(rdhIR);
    static bool firstCall = true;

    printf("%s HBF%4d (TF%3d/HB%3d) Sz:%4d| HB Orbit/BC :%4d/%4d Trigger:(0x%08x) %s Packet: %3d Page: %3d Stop: %d\n",
           rdh.stop ? "Close" : "Open ", hbfID, tfhb.first, tfhb.second, rdh.memorySize, rdhIR.orbit, rdhIR.bc,
           int(rdh.triggerType), trig.to_string().c_str(), rdh.packetCounter, int(rdh.pageCnt), int(rdh.stop));
    bool sox = (rdh.triggerType & o2::trigger::SOC || rdh.triggerType & o2::trigger::SOT);

    if (rdh.stop) {
      nHBFClose++;
    } else {
      nHBFOpen++;
      if (rdh.triggerType & o2::trigger::TF) {
        nTF++;
      }
      if (rdh.triggerType & (o2::trigger::ORBIT | o2::trigger::HB)) {
        nHBF++;
      }
      if (empty) {
        nHBFEmpty++;
      }
      BOOST_CHECK(firstCall == sox);
      firstCall = false;
    }
  };

  bool flagSOX = true; // the 1st RDH must provide readout mode: SOT or SOC

  for (int i = 0; i < nIRs; i++) {
    int nHBF = sampler.fillHBIRvector(HBIRVec, irFrom, irs[i]);
    irFrom = irs[i] + 1;

    // nHBF-1 HBframes don't have data, we need to create empty HBFs for them
    if (nHBF) {
      if (rdh.stop) { // do we need to close previous HBF?
        flushRDH();
      }
      for (int j = 0; j < nHBF - 1; j++) {
        rdhIR = HBIRVec[j];
        rdh = sampler.createRDH<RDH>(rdhIR);
        // dress rdh with cruID/FEE/Link ID ...
        rdh.packetCounter = packetCounter++;
        rdh.memorySize = sizeof(rdh);
        rdh.offsetToNext = sizeof(rdh);

        if (flagSOX) {
          rdh.triggerType |= useContinuous ? o2::trigger::SOC : o2::trigger::SOT;
          flagSOX = false;
        }
        flushRDH(); // open empty HBH
        rdh.packetCounter = packetCounter++;
        rdh.stop = 0x1;
        rdh.pageCnt++;
        flushRDH(); // close empty HBF
      }

      rdhIR = HBIRVec.back();
      rdh = sampler.createRDH<RDH>(rdhIR);
      if (flagSOX) {
        rdh.triggerType |= useContinuous ? o2::trigger::SOC : o2::trigger::SOT;
        flagSOX = false;
      }
      rdh.packetCounter = packetCounter++;
      rdh.memorySize = sizeof(rdh) + 16 + gRandom->Integer(8192 - sizeof(rdh) - 16); // random payload
      rdh.offsetToNext = rdh.memorySize;
      flushRDH(); // open non-empty HBH
      rdh.packetCounter = packetCounter++;
      rdh.stop = 0x1; // flag that it should be closed
      rdh.pageCnt++;
    }
    // flush payload
    printf("Flush payload for Orbit/BC %4d/%d\n", irs[i].orbit, irs[i].bc);
  }
  // close last packet
  if (rdh.stop) { // do we need to close previous HBF?
    flushRDH();
  } else {
    BOOST_CHECK(false); // lost closing RDH?
  }

  // the TF must be completed, generate HBF till the end of the current TF
  int tf = sampler.getTF(irs.back());
  auto lastHBIR = sampler.getIRTF(tf + 1) - 1; // last IR of the current TF
  sampler.fillHBIRvector(HBIRVec, irs.back(), lastHBIR);
  for (const auto& ir : HBIRVec) {
    rdh = sampler.createRDH<RDH>(rdhIR);
    // dress rdh with cruID/FEE/Link ID ...
    rdh.packetCounter = packetCounter++;
    rdh.memorySize = sizeof(rdh);
    rdh.offsetToNext = sizeof(rdh);

    flushRDH(); // open empty HBH
    rdh.stop = 0x1;
    rdh.pageCnt++;
    flushRDH(); // close empty HBF
  }

  printf("\nN_TF=%d, N_HBF=%d (%d empty), Opened %d / Closed %d\n", nTF, nHBF, nHBFEmpty, nHBFOpen, nHBFClose);
  BOOST_CHECK(nHBF > nHBFEmpty);
  BOOST_CHECK(nTF > 0);
  BOOST_CHECK(nHBFOpen == nHBFClose);
  BOOST_CHECK(nHBF == nTF * sampler.getNOrbitsPerTF()); // make sure all TFs are complete
}
} // namespace o2
