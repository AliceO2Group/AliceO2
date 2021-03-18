// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DetectorsCommonDataFormats/NameConf.h"
#include "MCHCTF/CTFCoder.h"
#include "DataFormatsMCH/CTF.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "MCHBase/Digit.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <cstring>

using namespace o2::mch;

BOOST_AUTO_TEST_CASE(CTFTest)
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
      uint16_t nsamp = gRandom->Integer(1025);
      auto& d = digs.emplace_back(detID, padID, adc, tfTime, nsamp);
      d.setSaturated(gRandom->Rndm() > 0.9);
    }
    rofs.emplace_back(ir, start, nch);
  }

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder;
    coder.encode(vec, rofs, digs); // compress
  }
  sw.Stop();
  LOG(INFO) << "Compressed in " << sw.CpuTime() << " s";

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
    LOG(INFO) << "Wrote to tree in " << sw.CpuTime() << " s";
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
    LOG(INFO) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<ROFRecord> rofsD;
  std::vector<Digit> digsD;

  sw.Start();
  const auto ctfImage = o2::mch::CTF::getImage(vec.data());
  {
    CTFCoder coder;
    coder.decode(ctfImage, rofsD, digsD); // decompress
  }
  sw.Stop();
  LOG(INFO) << "Decompressed in " << sw.CpuTime() << " s";

  BOOST_CHECK(rofsD.size() == rofs.size());
  BOOST_CHECK(digsD.size() == digs.size());
  LOG(INFO) << " BOOST_CHECK rofsD.size() " << rofsD.size() << " rofs.size() " << rofs.size()
            << " BOOST_CHECK(digsD.size() " << digsD.size() << " digs.size()) " << digs.size();

  for (size_t i = 0; i < rofs.size(); i++) {
    const auto& dor = rofs[i];
    const auto& ddc = rofsD[i];
    LOG(DEBUG) << " Orig.ROFRecord " << i << " " << dor.getBCData() << " " << dor.getFirstIdx() << " " << dor.getNEntries();
    LOG(DEBUG) << " Deco.ROFRecord " << i << " " << ddc.getBCData() << " " << ddc.getFirstIdx() << " " << ddc.getNEntries();

    BOOST_CHECK(dor.getBCData() == ddc.getBCData());
    BOOST_CHECK(dor.getFirstIdx() == ddc.getFirstIdx());
    BOOST_CHECK(dor.getNEntries() == ddc.getNEntries());
  }

  for (size_t i = 0; i < digs.size(); i++) {
    const auto& cor = digs[i];
    const auto& cdc = digsD[i];
    BOOST_CHECK(cor.getDetID() == cdc.getDetID());
    BOOST_CHECK(cor.getPadID() == cdc.getPadID());
    BOOST_CHECK(cor.getTime() == cdc.getTime());
    BOOST_CHECK(cor.nofSamples() == cdc.nofSamples());
    BOOST_CHECK(cor.getADC() == cdc.getADC());
  }
}
