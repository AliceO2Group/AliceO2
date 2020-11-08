// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test EMCCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DetectorsCommonDataFormats/NameConf.h"
#include "EMCALReconstruction/CTFCoder.h"
#include "DataFormatsEMCAL/CTF.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <cstring>

using namespace o2::emcal;

BOOST_AUTO_TEST_CASE(CTFTest)
{
  std::vector<TriggerRecord> triggers;
  std::vector<Cell> cells;
  //  gSystem->Load("libO2DetectorsCommonDataFormats.so");
  TStopwatch sw;
  sw.Start();
  o2::InteractionRecord ir(0, 0);
  for (int irof = 0; irof < 1000; irof++) {
    ir += 1 + gRandom->Integer(200);

    auto start = cells.size();
    short tower = gRandom->Poisson(10);
    while (tower < 17665) {
      float timeCell = gRandom->Rndm() * 1500 - 600.;
      float en = gRandom->Rndm() * 250.;
      int stat = gRandom->Integer(5);
      cells.emplace_back(tower, en, timeCell, (ChannelType_t)stat);
      tower += 1 + gRandom->Integer(100);
    }
    triggers.emplace_back(ir, start, cells.size() - start);
  }

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder;
    coder.encode(vec, triggers, cells); // compress
  }
  sw.Stop();
  LOG(INFO) << "Compressed in " << sw.CpuTime() << " s";

  // writing
  {
    sw.Start();
    auto* ctfImage = o2::emcal::CTF::get(vec.data());
    TFile flOut("test_ctf_emcal.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "EMC");
    ctfTree.Write();
    sw.Stop();
    LOG(INFO) << "Wrote to tree in " << sw.CpuTime() << " s";
  }

  // reading
  vec.clear();
  {
    sw.Start();
    TFile flIn("test_ctf_emcal.root");
    std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    BOOST_CHECK(tree);
    o2::emcal::CTF::readFromTree(vec, *(tree.get()), "EMC");
    sw.Stop();
    LOG(INFO) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<TriggerRecord> triggersD;
  std::vector<Cell> cellsD;

  sw.Start();
  const auto ctfImage = o2::emcal::CTF::getImage(vec.data());
  {
    CTFCoder coder;
    coder.decode(ctfImage, triggersD, cellsD); // decompress
  }
  sw.Stop();
  LOG(INFO) << "Decompressed in " << sw.CpuTime() << " s";

  BOOST_CHECK(triggersD.size() == triggers.size());
  BOOST_CHECK(cellsD.size() == cells.size());
  LOG(INFO) << " BOOST_CHECK triggersD.size() " << triggersD.size() << " triggers.size() " << triggers.size()
            << " BOOST_CHECK(cellsD.size() " << cellsD.size() << " cells.size()) " << cells.size();

  for (size_t i = 0; i < triggers.size(); i++) {
    const auto& dor = triggers[i];
    const auto& ddc = triggersD[i];
    LOG(DEBUG) << " Orig.TriggerRecord " << i << " " << dor.getBCData() << " " << dor.getFirstEntry() << " " << dor.getNumberOfObjects();
    LOG(DEBUG) << " Deco.TriggerRecord " << i << " " << ddc.getBCData() << " " << ddc.getFirstEntry() << " " << ddc.getNumberOfObjects();

    BOOST_CHECK(dor.getBCData() == ddc.getBCData());
    BOOST_CHECK(dor.getNumberOfObjects() == ddc.getNumberOfObjects());
    BOOST_CHECK(dor.getFirstEntry() == dor.getFirstEntry());
  }

  for (size_t i = 0; i < cells.size(); i++) {
    const auto& cor = cells[i];
    const auto& cdc = cellsD[i];
    BOOST_CHECK(cor.getPackedTowerID() == cdc.getPackedTowerID());
    BOOST_CHECK(cor.getPackedTime() == cdc.getPackedTime());
    BOOST_CHECK(cor.getPackedEnergy() == cdc.getPackedEnergy());
    BOOST_CHECK(cor.getPackedCellStatus() == cdc.getPackedCellStatus());
  }
}
