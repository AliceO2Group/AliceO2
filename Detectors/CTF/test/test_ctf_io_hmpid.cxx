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

#define BOOST_TEST_MODULE Test HMPIDCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "CommonUtils/NameConf.h"
#include "HMPIDReconstruction/CTFCoder.h"
#include "DataFormatsHMP/CTF.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <cstring>

using namespace o2::hmpid;

BOOST_AUTO_TEST_CASE(CTFTest)
{
  std::vector<Trigger> triggers;
  std::vector<Digit> digits;
  TStopwatch sw;
  sw.Start();
  o2::InteractionRecord ir(0, 0);
  Digit clu;
  for (int irof = 0; irof < 1000; irof++) {
    ir += 1 + gRandom->Integer(200);

    auto start = digits.size();
    uint8_t chID = 0;
    int n = 0;
    while ((chID += gRandom->Integer(10)) < 0xff) {
      uint16_t q = gRandom->Integer(0xffff);
      uint8_t ph = gRandom->Integer(0xff);
      uint8_t x = gRandom->Integer(0xff);
      uint8_t y = gRandom->Integer(0xff);
      digits.emplace_back(chID, ph, x, y, q);
    }
    triggers.emplace_back(ir, start, digits.size() - start);
  }

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Encoder);
    coder.encode(vec, triggers, digits); // compress
  }
  sw.Stop();
  LOG(info) << "Compressed in " << sw.CpuTime() << " s";

  // writing
  {
    sw.Start();
    auto* ctfImage = o2::hmpid::CTF::get(vec.data());
    TFile flOut("test_ctf_hmpid.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "HMP");
    ctfTree.Write();
    sw.Stop();
    LOG(info) << "Wrote to tree in " << sw.CpuTime() << " s";
  }

  // reading
  vec.clear();
  LOG(info) << "Start reading from tree ";
  {
    sw.Start();
    TFile flIn("test_ctf_hmpid.root");
    std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    BOOST_CHECK(tree);
    o2::hmpid::CTF::readFromTree(vec, *(tree.get()), "HMP");
    sw.Stop();
    LOG(info) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<Trigger> triggersD;
  std::vector<Digit> digitsD;

  sw.Start();
  const auto ctfImage = o2::hmpid::CTF::getImage(vec.data());
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Decoder);
    coder.decode(ctfImage, triggersD, digitsD); // decompress
  }
  sw.Stop();
  LOG(info) << "Decompressed in " << sw.CpuTime() << " s";

  BOOST_CHECK(triggersD.size() == triggers.size());
  BOOST_CHECK(digitsD.size() == digits.size());

  BOOST_TEST(triggersD == triggers, boost::test_tools::per_element());
  BOOST_TEST(digitsD == digits, boost::test_tools::per_element());
}
