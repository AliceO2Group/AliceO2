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

#define BOOST_TEST_MODULE Test TRDCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/dataset.hpp>
#include "CommonUtils/NameConf.h"
#include "TRDReconstruction/CTFCoder.h"
#include "DataFormatsTRD/CTF.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <cstring>

using namespace o2::trd;
namespace boost_data = boost::unit_test::data;

inline std::vector<o2::ctf::ANSHeader> ANSVersions{o2::ctf::ANSVersionCompat, o2::ctf::ANSVersion1};

BOOST_DATA_TEST_CASE(CTFTest, boost_data::make(ANSVersions), ansVersion)
{
  std::vector<TriggerRecord> triggers;
  std::vector<Tracklet64> tracklets;
  std::vector<Digit> digits;

  TStopwatch sw;
  sw.Start();
  o2::InteractionRecord ir(0, 0);
  constexpr int NCID = 540, NHCID = 2 * NCID;
  constexpr uint32_t formatTrk = 5;
  ArrayADC adc;

  for (int irof = 0; irof < 200; irof++) {
    ir += 1 + gRandom->Integer(600);
    bool doDigits = gRandom->Rndm() > 0.8;

    auto startTrk = tracklets.size();
    auto startDig = digits.size();
    int cid = 0;
    while ((cid += gRandom->Poisson(5)) < NHCID) {
      int hcid = cid / 2;
      int nTrk = gRandom->Poisson(3);
      int nDig = doDigits ? nTrk * 5 * (1. + gRandom->Rndm()) : 0;

      for (int i = nTrk; i--;) {
        tracklets.emplace_back(formatTrk, hcid, gRandom->Integer(0x1 << 4), gRandom->Integer(0x1 << 2),
                               gRandom->Integer(0x1 << 11), gRandom->Integer(0x1 << 8), gRandom->Integer(0x1 << 24));
      }
      for (int i = nDig; i--;) {
        auto& dig = digits.emplace_back(cid, gRandom->Integer(0x1 << 8), gRandom->Integer(0x1 << 8), gRandom->Integer(0x1 << 8));
        for (int j = constants::TIMEBINS; j--;) {
          adc[j] = gRandom->Integer(0x1 << 16);
        }
        dig.setADC(adc);
      }
    }

    triggers.emplace_back(ir, startDig, digits.size() - startDig, startTrk, tracklets.size() - startTrk);
  }

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Encoder);
    coder.setANSVersion(ansVersion);
    coder.encode(vec, triggers, tracklets, digits); // compress
  }
  sw.Stop();
  LOG(info) << "Compressed in " << sw.CpuTime() << " s";

  // writing
  {
    sw.Start();
    auto* ctfImage = o2::trd::CTF::get(vec.data());
    TFile flOut("test_ctf_trd.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "TRD");
    ctfTree.Write();
    sw.Stop();
    LOG(info) << "Wrote to tree in " << sw.CpuTime() << " s";
  }

  // reading
  vec.clear();
  LOG(info) << "Start reading from tree ";
  {
    sw.Start();
    TFile flIn("test_ctf_trd.root");
    std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    BOOST_CHECK(tree);
    o2::trd::CTF::readFromTree(vec, *(tree.get()), "TRD");
    sw.Stop();
    LOG(info) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<TriggerRecord> triggersD;
  std::vector<Tracklet64> trackletsD;
  std::vector<Digit> digitsD;

  sw.Start();
  const auto ctfImage = o2::trd::CTF::getImage(vec.data());
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Decoder);
    coder.decode(ctfImage, triggersD, trackletsD, digitsD); // decompress
  }
  sw.Stop();
  LOG(info) << "Decompressed in " << sw.CpuTime() << " s";

  BOOST_CHECK(triggersD.size() == triggers.size());
  BOOST_CHECK(trackletsD.size() == tracklets.size());
  BOOST_CHECK(digitsD.size() == digitsD.size());

  BOOST_TEST(triggersD == triggers, boost::test_tools::per_element());
  BOOST_TEST(trackletsD == tracklets, boost::test_tools::per_element());
  BOOST_TEST(digitsD == digits, boost::test_tools::per_element());
}
