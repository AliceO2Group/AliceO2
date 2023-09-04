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

#define BOOST_TEST_MODULE Test PHSCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/dataset.hpp>
#include "CommonUtils/NameConf.h"
#include "PHOSReconstruction/CTFCoder.h"
#include "DataFormatsPHOS/CTF.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <cstring>

using namespace o2::phos;
namespace boost_data = boost::unit_test::data;

inline std::vector<o2::ctf::ANSHeader> ANSVersions{o2::ctf::ANSVersionCompat, o2::ctf::ANSVersion1};

BOOST_DATA_TEST_CASE(CTFTest, boost_data::make(ANSVersions), ansVersion)
{
  std::vector<TriggerRecord> triggers;
  std::vector<Cell> cells;
  TStopwatch sw;
  sw.Start();
  o2::InteractionRecord ir(0, 0);
  for (int irof = 0; irof < 1000; irof++) {
    ir += 1 + gRandom->Integer(200);

    auto start = cells.size();
    int n = 1 + gRandom->Poisson(100);
    for (int i = n; i--;) {
      ChannelType_t tp = gRandom->Rndm() > 0.5 ? (gRandom->Rndm() > 0.5 ? TRU2x2 : TRU4x4) : (gRandom->Rndm() > 0.5 ? HIGH_GAIN : LOW_GAIN);
      uint16_t id = (tp == TRU2x2 || tp == TRU4x4) ? 3000 : gRandom->Integer(kNmaxCell);
      float timeCell = gRandom->Rndm() * 3.00e-07 - 0.3e-9;
      float en = gRandom->Rndm() * 160.;
      cells.emplace_back(id, en, timeCell, tp);
    }
    triggers.emplace_back(ir, start, cells.size() - start);
  }

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Encoder);
    coder.setANSVersion(ansVersion);
    coder.encode(vec, triggers, cells); // compress
  }
  sw.Stop();
  LOG(info) << "Compressed in " << sw.CpuTime() << " s";

  // writing
  {
    sw.Start();
    auto* ctfImage = o2::phos::CTF::get(vec.data());
    TFile flOut("test_ctf_phos.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "PHS");
    ctfTree.Write();
    sw.Stop();
    LOG(info) << "Wrote to tree in " << sw.CpuTime() << " s";
  }

  // reading
  vec.clear();
  {
    sw.Start();
    TFile flIn("test_ctf_phos.root");
    std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    BOOST_CHECK(tree);
    o2::phos::CTF::readFromTree(vec, *(tree.get()), "PHS");
    sw.Stop();
    LOG(info) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<TriggerRecord> triggersD;
  std::vector<Cell> cellsD;

  sw.Start();
  const auto ctfImage = o2::phos::CTF::getImage(vec.data());
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Decoder);
    coder.decode(ctfImage, triggersD, cellsD); // decompress
  }
  sw.Stop();
  LOG(info) << "Decompressed in " << sw.CpuTime() << " s";

  BOOST_CHECK(triggersD.size() == triggers.size());
  BOOST_CHECK(cellsD.size() == cells.size());
  LOG(info) << " BOOST_CHECK triggersD.size() " << triggersD.size() << " triggers.size() " << triggers.size()
            << " BOOST_CHECK(cellsD.size() " << cellsD.size() << " cells.size()) " << cells.size();

  for (size_t i = 0; i < triggers.size(); i++) {
    const auto& dor = triggers[i];
    const auto& ddc = triggersD[i];
    LOG(debug) << " Orig.TriggerRecord " << i << " " << dor.getBCData() << " " << dor.getFirstEntry() << " " << dor.getNumberOfObjects();
    LOG(debug) << " Deco.TriggerRecord " << i << " " << ddc.getBCData() << " " << ddc.getFirstEntry() << " " << ddc.getNumberOfObjects();

    BOOST_CHECK(dor.getBCData() == ddc.getBCData());
    BOOST_CHECK(dor.getNumberOfObjects() == ddc.getNumberOfObjects());
    BOOST_CHECK(dor.getFirstEntry() == dor.getFirstEntry());
  }

  for (size_t i = 0; i < cells.size(); i++) {
    const auto& cor = cells[i];
    const auto& cdc = cellsD[i];
    BOOST_CHECK(cor.getPackedID() == cdc.getPackedID());
    BOOST_CHECK(cor.getPackedTime() == cdc.getPackedTime());
    BOOST_CHECK(cor.getPackedEnergy() == cdc.getPackedEnergy());
    BOOST_CHECK(cor.getPackedCellStatus() == cdc.getPackedCellStatus());
  }
}
