// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test FV0CTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DetectorsCommonDataFormats/NameConf.h"
#include "FV0Reconstruction/CTFCoder.h"
#include "FV0Base/Constants.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <cstring>

using namespace o2::fv0;

BOOST_AUTO_TEST_CASE(CTFTest)
{
  std::vector<BCData> digits;
  std::vector<ChannelData> channels;
  Triggers trigger; // TODO: Actual values are not set

  TStopwatch sw;
  sw.Start();
  o2::InteractionRecord ir(0, 0);

  constexpr int MAXChan = Constants::nChannelsPerPm * Constants::nPms; // RSFIXME is this correct ?
  for (int idig = 0; idig < 1000; idig++) {
    ir += 1 + gRandom->Integer(200);
    uint8_t ich = gRandom->Poisson(10);
    auto start = channels.size();
    while (ich < MAXChan) {
      int16_t t = -2048 + gRandom->Integer(2048 * 2);
      uint16_t q = gRandom->Integer(4096);
      channels.emplace_back(ich, t, q);
      ich += 1 + gRandom->Poisson(10);
    }
    auto end = channels.size();

    digits.emplace_back(start, end - start, ir, trigger);
  }

  LOG(INFO) << "Generated " << channels.size() << " channels in " << digits.size() << " digits " << sw.CpuTime() << " s";

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder;
    coder.encode(vec, digits, channels); // compress
  }
  sw.Stop();
  LOG(INFO) << "Compressed in " << sw.CpuTime() << " s";

  // writing
  {
    sw.Start();
    TFile flOut("test_ctf_fv0.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    auto* ctfImage = o2::fv0::CTF::get(vec.data());
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "FV0");
    ctfTree.Write();
    sw.Stop();
    LOG(INFO) << "Wrote to tree in " << sw.CpuTime() << " s";
  }

  // reading
  vec.clear();
  {
    sw.Start();
    TFile flIn("test_ctf_fv0.root");
    std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    BOOST_CHECK(tree);
    o2::fv0::CTF::readFromTree(vec, *(tree.get()), "FV0");
    sw.Stop();
    LOG(INFO) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<BCData> digitsD;
  std::vector<ChannelData> channelsD;

  sw.Start();
  const auto ctfImage = o2::fv0::CTF::getImage(vec.data());
  {
    CTFCoder coder;
    coder.decode(ctfImage, digitsD, channelsD); // decompress
  }
  sw.Stop();
  LOG(INFO) << "Decompressed in " << sw.CpuTime() << " s";

  BOOST_CHECK(digitsD.size() == digits.size());
  BOOST_CHECK(channelsD.size() == channels.size());
  LOG(INFO) << "  BOOST_CHECK digitsD.size() " << digitsD.size() << " digits.size() " << digits.size() << " BOOST_CHECK(channelsD.size()  " << channelsD.size() << " channels.size()) " << channels.size();

  for (int i = digits.size(); i--;) {
    const auto& dor = digits[i];
    const auto& ddc = digitsD[i];
    BOOST_CHECK(dor.ir == ddc.ir);
    BOOST_CHECK(dor.ref == ddc.ref);
  }
  for (int i = channels.size(); i--;) {
    const auto& cor = channels[i];
    const auto& cdc = channelsD[i];
    BOOST_CHECK(cor.pmtNumber == cdc.pmtNumber);
    BOOST_CHECK(cor.time == cdc.time);
    BOOST_CHECK(cor.chargeAdc == cdc.chargeAdc);
  }
}
