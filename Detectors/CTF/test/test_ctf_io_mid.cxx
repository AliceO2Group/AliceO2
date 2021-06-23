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

#define BOOST_TEST_MODULE Test MIDCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DetectorsCommonDataFormats/NameConf.h"
#include "MIDCTF/CTFCoder.h"
#include "DataFormatsMID/CTF.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <cstring>

using namespace o2::mid;

BOOST_AUTO_TEST_CASE(CTFTest)
{
  std::vector<ROFRecord> rofs;
  std::vector<ColumnData> cols;
  // RS: don't understand why, but this library is not loaded automatically, although the dependencies are clearly
  // indicated. What it more weird is that for similar tests of other detectors the library is loaded!
  // Absence of the library leads to complains about the StreamerInfo and eventually segm.faul when appending the
  // CTH to the tree. For this reason I am loading it here manually
  gSystem->Load("libO2DetectorsCommonDataFormats");
  TStopwatch sw;
  sw.Start();
  o2::InteractionRecord ir(0, 0);
  std::array<uint16_t, 5> pattern;
  for (int irof = 0; irof < 1000; irof++) {
    ir += 1 + gRandom->Integer(200);
    uint8_t nch = 0, evtyp = gRandom->Integer(3);
    while (nch == 0) {
      nch = gRandom->Poisson(10);
    }
    auto start = cols.size();
    for (int ich = 0; ich < nch; ich++) {
      uint8_t deId = gRandom->Integer(128);
      uint8_t columnId = gRandom->Integer(128);
      for (int i = 0; i < 5; i++) {
        pattern[i] = gRandom->Integer(0x7fff);
      }
      cols.emplace_back(ColumnData{deId, columnId, pattern});
    }
    rofs.emplace_back(ROFRecord{ir, EventType(evtyp), start, cols.size() - start});
  }

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder;
    coder.encode(vec, rofs, cols); // compress
  }
  sw.Stop();
  LOG(INFO) << "Compressed in " << sw.CpuTime() << " s";

  // writing
  {
    sw.Start();
    auto* ctfImage = o2::mid::CTF::get(vec.data());
    TFile flOut("test_ctf_mid.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "MID");
    ctfTree.Write();
    sw.Stop();
    LOG(INFO) << "Wrote to tree in " << sw.CpuTime() << " s";
  }

  // reading
  vec.clear();
  {
    sw.Start();
    TFile flIn("test_ctf_mid.root");
    std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    BOOST_CHECK(tree);
    o2::mid::CTF::readFromTree(vec, *(tree.get()), "MID");
    sw.Stop();
    LOG(INFO) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<ROFRecord> rofsD;
  std::vector<ColumnData> colsD;

  sw.Start();
  const auto ctfImage = o2::mid::CTF::getImage(vec.data());
  {
    CTFCoder coder;
    coder.decode(ctfImage, rofsD, colsD); // decompress
  }
  sw.Stop();
  LOG(INFO) << "Decompressed in " << sw.CpuTime() << " s";

  BOOST_CHECK(rofsD.size() == rofs.size());
  BOOST_CHECK(colsD.size() == cols.size());
  LOG(INFO) << " BOOST_CHECK rofsD.size() " << rofsD.size() << " rofs.size() " << rofs.size()
            << " BOOST_CHECK(colsD.size() " << colsD.size() << " cols.size()) " << cols.size();

  for (size_t i = 0; i < rofs.size(); i++) {
    const auto& dor = rofs[i];
    const auto& ddc = rofsD[i];
    LOG(DEBUG) << " Orig.ROFRecord " << i << " " << dor.interactionRecord << " " << dor.firstEntry << " " << dor.nEntries;
    LOG(DEBUG) << " Deco.ROFRecord " << i << " " << ddc.interactionRecord << " " << ddc.firstEntry << " " << ddc.nEntries;

    BOOST_CHECK(dor.interactionRecord == ddc.interactionRecord);
    BOOST_CHECK(dor.firstEntry == ddc.firstEntry);
    BOOST_CHECK(dor.nEntries == dor.nEntries);
  }

  for (size_t i = 0; i < cols.size(); i++) {
    const auto& cor = cols[i];
    const auto& cdc = colsD[i];
    BOOST_CHECK(cor.deId == cdc.deId);
    BOOST_CHECK(cor.columnId == cdc.columnId);
    for (int j = 0; j < 5; j++) {
      BOOST_CHECK(cor.patterns[j] == cdc.patterns[j]);
      LOG(DEBUG) << "col " << i << " pat " << j << " : " << cor.patterns[j] << " : " << cdc.patterns[j];
    }
  }
}
