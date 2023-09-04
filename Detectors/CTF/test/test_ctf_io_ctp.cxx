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

#define BOOST_TEST_MODULE Test CTPCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/dataset.hpp>
#include "CommonUtils/NameConf.h"
#include "CTPReconstruction/CTFCoder.h"
#include "DataFormatsCTP/CTF.h"
#include "DataFormatsCTP/Digits.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <cstring>
#include <random>

using namespace o2::ctp;
namespace boost_data = boost::unit_test::data;

inline std::vector<o2::ctf::ANSHeader> ANSVersions{o2::ctf::ANSVersionCompat, o2::ctf::ANSVersion1};

BOOST_DATA_TEST_CASE(CTFTest, boost_data::make(ANSVersions), ansVersion)
{
  std::vector<CTPDigit> digits;
  TStopwatch sw;
  sw.Start();
  o2::InteractionRecord ir0(3, 5), ir(ir0);

  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<unsigned long long> distr;
  for (int itrg = 0; itrg < 1000; itrg++) {
    ir += 1 + distr(eng) % 200;
    auto& dig = digits.emplace_back();
    dig.intRecord = ir;
    dig.CTPInputMask |= distr(eng);
    dig.CTPClassMask |= distr(eng);
  }
  LumiInfo lumi{digits.front().intRecord.orbit, 30, 12345};

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Encoder);
    coder.setANSVersion(ansVersion);
    coder.encode(vec, digits, lumi); // compress
  }
  sw.Stop();
  LOG(info) << "Compressed in " << sw.CpuTime() << " s";

  // writing
  {
    sw.Start();
    auto* ctfImage = o2::ctp::CTF::get(vec.data());
    TFile flOut("test_ctf_ctp.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "CTP");
    ctfTree.Write();
    sw.Stop();
    LOG(info) << "Wrote to tree in " << sw.CpuTime() << " s";
  }

  // reading
  vec.clear();
  {
    sw.Start();
    TFile flIn("test_ctf_ctp.root");
    std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    BOOST_CHECK(tree);
    o2::ctp::CTF::readFromTree(vec, *(tree.get()), "CTP");
    sw.Stop();
    LOG(info) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<CTPDigit> digitsD;
  LumiInfo lumiD;
  sw.Start();
  const auto ctfImage = o2::ctp::CTF::getImage(vec.data());
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Decoder);
    coder.decode(ctfImage, digitsD, lumiD); // decompress
  }
  sw.Stop();
  LOG(info) << "Decompressed in " << sw.CpuTime() << " s";

  LOG(info) << " BOOST_CHECK(digitsD.size() " << digitsD.size() << " digigits.size()) " << digits.size();

  BOOST_TEST(digits == digitsD, boost::test_tools::per_element());
  BOOST_CHECK(lumiD.orbit == lumi.orbit && lumiD.nHBFCounted == lumi.nHBFCounted && lumiD.orbit == lumi.orbit);
}
