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

#define BOOST_TEST_MODULE Test EMCCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/dataset.hpp>
#include "CommonUtils/NameConf.h"
#include "DataFormatsEMCAL/Constants.h"
#include "EMCALReconstruction/CTFCoder.h"
#include "DataFormatsEMCAL/CTF.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <cstring>

using namespace o2::emcal;
namespace boost_data = boost::unit_test::data;

inline std::vector<o2::ctf::ANSHeader> ANSVersions{o2::ctf::ANSVersionCompat, o2::ctf::ANSVersion1};

BOOST_DATA_TEST_CASE(CTFTest, boost_data::make(ANSVersions), ansVersion)
{
  std::vector<TriggerRecord> triggers;
  std::vector<Cell> cells;
  //  gSystem->Load("libO2DetectorsCommonDataFormats");
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
      // In case of cell type 3 cases must be distinguished (FEE, LEDMON, TRU)
      // In case the cell is a FEE cell the cell type is correlated with the energy
      int readoutsource = gRandom->Integer(3); //
      ChannelType_t chantype = ChannelType_t::HIGH_GAIN;
      switch (readoutsource) {
        case 0: {
          // Cell is a FEE cell, determine cell type according to HGLG transition
          const auto ENHGLG = o2::emcal::constants::EMCAL_HGLGTRANSITION * o2::emcal::constants::EMCAL_ADCENERGY;
          if (en >= ENHGLG) {
            chantype = ChannelType_t::LOW_GAIN;
          } else {
            chantype = ChannelType_t::HIGH_GAIN;
          }
          break;
        }
        case 1:
          chantype = ChannelType_t::LEDMON;
          break;
        case 2:
          chantype = ChannelType_t::TRU;
          break;

        default:
          std::cerr << "Unknown type" << std::endl;
          break;
      }
      cells.emplace_back(tower, en, timeCell, chantype);
      tower += 1 + gRandom->Integer(100);
    }
    uint32_t trigBits = gRandom->Integer(0xFFFFFFFF); // will be converted internally to uint16_t by the coder
    triggers.emplace_back(ir, trigBits, start, cells.size() - start);
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
    auto* ctfImage = o2::emcal::CTF::get(vec.data());
    TFile flOut("test_ctf_emcal.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "EMC");
    ctfTree.Write();
    sw.Stop();
    LOG(info) << "Wrote to tree in " << sw.CpuTime() << " s";
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
    LOG(info) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<TriggerRecord> triggersD;
  std::vector<Cell> cellsD;

  sw.Start();
  const auto ctfImage = o2::emcal::CTF::getImage(vec.data());
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Decoder);
    coder.decode(ctfImage, triggersD, cellsD); // decompress
  }
  sw.Stop();
  LOG(info) << "Decompressed in " << sw.CpuTime() << " s";

  BOOST_CHECK_EQUAL(triggersD.size(), triggers.size());
  BOOST_CHECK_EQUAL(cellsD.size(), cells.size());
  LOG(info) << " BOOST_CHECK triggersD.size() " << triggersD.size() << " triggers.size() " << triggers.size()
            << " BOOST_CHECK(cellsD.size() " << cellsD.size() << " cells.size()) " << cells.size();

  for (size_t i = 0; i < triggers.size(); i++) {
    const auto& dor = triggers[i];
    const auto& ddc = triggersD[i];
    LOG(debug) << " Orig.TriggerRecord " << i << " " << dor.getBCData() << " " << dor.getFirstEntry() << " " << dor.getNumberOfObjects();
    LOG(debug) << " Deco.TriggerRecord " << i << " " << ddc.getBCData() << " " << ddc.getFirstEntry() << " " << ddc.getNumberOfObjects();

    BOOST_CHECK_EQUAL(dor.getBCData(), ddc.getBCData());
    BOOST_CHECK_EQUAL(dor.getNumberOfObjects(), ddc.getNumberOfObjects());
    BOOST_CHECK_EQUAL(dor.getFirstEntry(), ddc.getFirstEntry());
    BOOST_CHECK_EQUAL(dor.getTriggerBitsCompressed(), ddc.getTriggerBitsCompressed()); // Need to be compared to the filtered trigger bit set
    // Check for the function getTriggerBits
    // As the compessed version has trigger bits discarded,
    // reference must be constructed again from compressed
    // trigger bits. Otherwise the reconstructed object is
    // compared to the uncompressed version and the test will
    // obviously fail due to the bits which are removed.
    // Therefore a copy is needed to modify the trigger bits
    // storing only the compressed one
    auto triggerbittest = triggers[i];
    triggerbittest.setTriggerBitsCompressed(triggerbittest.getTriggerBitsCompressed());
    BOOST_CHECK_EQUAL(triggerbittest.getTriggerBits(), ddc.getTriggerBits());
  }

  for (size_t i = 0; i < cells.size(); i++) {
    const auto& cor = cells[i];
    const auto& cdc = cellsD[i];
    BOOST_CHECK_EQUAL(cor.getTowerIDEncoded(), cdc.getTowerIDEncoded());
    BOOST_CHECK_EQUAL(cor.getTimeStampEncoded(), cdc.getTimeStampEncoded());
    BOOST_CHECK_EQUAL(cor.getEnergyEncoded(), cdc.getEnergyEncoded());
    BOOST_CHECK_EQUAL(cor.getCellTypeEncoded(), cdc.getCellTypeEncoded());
  }
}
