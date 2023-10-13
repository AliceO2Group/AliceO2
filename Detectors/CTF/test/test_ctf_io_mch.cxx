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

#define BOOST_TEST_MODULE Test MCHCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/dataset.hpp>
#include "CommonUtils/NameConf.h"
#include "MCHCTF/CTFCoder.h"
#include "DataFormatsMCH/CTF.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <cstring>

using namespace o2::mch;
namespace boost_data = boost::unit_test::data;

inline std::vector<o2::ctf::ANSHeader> ANSVersions{o2::ctf::ANSVersionCompat, o2::ctf::ANSVersion1};

BOOST_DATA_TEST_CASE(CTFTest, boost_data::make(ANSVersions), ansVersion)
{
  std::vector<ROFRecord> rofs;
  std::vector<Digit> digs;
  TStopwatch sw;
  sw.Start();
  o2::InteractionRecord ir0(3, 5), ir(ir0);

  for (int irof = 0; irof < 1000; irof++) {
    ir += 1 + gRandom->Integer(200);
    int nch = 0;
    while (nch == 0) {
      nch = gRandom->Poisson(20);
    }
    int start = digs.size();
    for (int ich = 0; ich < nch; ich++) {
      int16_t detID = 100 + gRandom->Integer(1025 - 100);
      int16_t padID = gRandom->Integer(28672);
      int32_t tfTime = ir.differenceInBC(ir0);
      uint32_t adc = gRandom->Integer(1024 * 1024);
      uint16_t nsamp = gRandom->Integer(1024);
      auto& d = digs.emplace_back(detID, padID, adc, tfTime, nsamp);
      bool sat = gRandom->Rndm() > 0.9;
      d.setSaturated(sat);
    }
    rofs.emplace_back(ir, start, nch);
  }

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Encoder);
    coder.setANSVersion(ansVersion);
    coder.encode(vec, rofs, digs); // compress
  }
  sw.Stop();
  LOG(info) << "Compressed in " << sw.CpuTime() << " s";

  // writing
  {
    sw.Start();
    auto* ctfImage = o2::mch::CTF::get(vec.data());
    TFile flOut("test_ctf_mch.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "MCH");
    ctfTree.Write();
    sw.Stop();
    LOG(info) << "Wrote to tree in " << sw.CpuTime() << " s";
  }

  // reading
  vec.clear();
  {
    sw.Start();
    TFile flIn("test_ctf_mch.root");
    std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    BOOST_CHECK(tree);
    o2::mch::CTF::readFromTree(vec, *(tree.get()), "MCH");
    sw.Stop();
    LOG(info) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<ROFRecord> rofsD;
  std::vector<Digit> digsD;

  sw.Start();
  const auto ctfImage = o2::mch::CTF::getImage(vec.data());
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Decoder);
    coder.decode(ctfImage, rofsD, digsD); // decompress
  }
  sw.Stop();
  LOG(info) << "Decompressed in " << sw.CpuTime() << " s";

  LOG(info) << " BOOST_CHECK rofsD.size() " << rofsD.size() << " rofs.size() " << rofs.size()
            << " BOOST_CHECK(digsD.size() " << digsD.size() << " digs.size()) " << digs.size();

  BOOST_TEST(rofs == rofsD, boost::test_tools::per_element());
  BOOST_TEST(digs == digsD, boost::test_tools::per_element());
}
