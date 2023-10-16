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

#define BOOST_TEST_MODULE Test ZDCCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "CommonUtils/NameConf.h"
#include "ZDCReconstruction/CTFCoder.h"
#include "DataFormatsZDC/CTF.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <cstring>

using namespace o2::zdc;

BOOST_AUTO_TEST_CASE(CTFTest)
{
  std::vector<BCData> bcdata;
  std::vector<ChannelData> chandata;
  std::vector<OrbitData> pedsdata;
  // RS: don't understand why, but this library is not loaded automatically, although the dependencies are clearly
  // indicated. What it more weird is that for similar tests of other detectors the library is loaded!
  // Absence of the library leads to complains about the StreamerInfo and eventually segm.faul when appending the
  // CTH to the tree. For this reason I am loading it here manually
  //  gSystem->Load("libO2DetectorsCommonDataFormats");
  TStopwatch sw;
  sw.Start();
  o2::InteractionRecord ir(0, 0);
  std::array<float, NTimeBinsPerBC> chanVals;

  // BCData and ChannelData
  for (int irof = 0; irof < 1000; irof++) {
    ir += 1 + gRandom->Integer(100); // randomly increaing BC

    uint32_t channPatt = 0, triggers = 0;
    int8_t ich = -1;
    int firstChEntry = chandata.size();
    while ((ich += 1 + gRandom->Poisson(2.)) < NDigiChannels) {
      channPatt |= 0x1 << ich;
      for (int i = 0; i < NTimeBinsPerBC; i++) {
        chanVals[i] = gRandom->Integer(0xffff);
      }
      if (gRandom->Rndm() > 0.4) {
        triggers |= 0x1 << ich;
      }
      chandata.emplace_back(ich, chanVals);
    }
    auto& bcd = bcdata.emplace_back(firstChEntry, chandata.size() - firstChEntry, ir, channPatt, triggers, gRandom->Integer(0xff));
    for (int im = 0; im < NModules; im++) {
      bcd.moduleTriggers[im] = gRandom->Rndm() > 0.7 ? gRandom->Integer((0x1 << 10) - 1) : 0;
    }
  }

  // OrbitData
  const auto &irFirst = bcdata.front().ir, irLast = bcdata.back().ir;
  o2::InteractionRecord irPed(o2::constants::lhc::LHCMaxBunches - 1, irFirst.orbit);
  int norbits = irLast.orbit - irFirst.orbit + 1;
  pedsdata.resize(norbits);
  for (int i = 0; i < norbits; i++) {
    pedsdata[i].ir = irPed;
    for (int ic = 0; ic < NChannels; ic++) {
      pedsdata[i].data[ic] = gRandom->Integer(0xffff);
      pedsdata[i].scaler[ic] = (i > 0 ? pedsdata[i].scaler[ic - 1] : 0) + gRandom->Integer(20);
    }
    irPed.orbit++;
  }

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Encoder);
    coder.encode(vec, bcdata, chandata, pedsdata); // compress
  }
  sw.Stop();
  LOG(info) << "Compressed in " << sw.CpuTime() << " s";

  // writing
  {
    sw.Start();
    auto* ctfImage = o2::zdc::CTF::get(vec.data());
    TFile flOut("test_ctf_zdc.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "ZDC");
    ctfTree.Write();
    sw.Stop();
    LOG(info) << "Wrote to tree in " << sw.CpuTime() << " s";
  }

  // reading
  vec.clear();
  {
    sw.Start();
    TFile flIn("test_ctf_zdc.root");
    std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    BOOST_CHECK(tree);
    o2::zdc::CTF::readFromTree(vec, *(tree.get()), "ZDC");
    sw.Stop();
    LOG(info) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<BCData> bcdataD;
  std::vector<ChannelData> chandataD;
  std::vector<OrbitData> pedsdataD;

  sw.Start();
  const auto ctfImage = o2::zdc::CTF::getImage(vec.data());
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Decoder);
    coder.decode(ctfImage, bcdataD, chandataD, pedsdataD); // decompress
  }
  sw.Stop();
  LOG(info) << "Decompressed in " << sw.CpuTime() << " s";

  LOG(info) << "Testing BCData: BOOST_CHECK bcdataD.size() " << bcdataD.size() << " bcdata.size() " << bcdata.size();
  BOOST_CHECK(bcdataD.size() == bcdata.size());
  for (size_t i = 0; i < bcdata.size(); i++) {
    bool cmpBCData = (bcdata[i].ir == bcdataD[i].ir &&
                      bcdata[i].ref == bcdataD[i].ref &&
                      bcdata[i].moduleTriggers == bcdataD[i].moduleTriggers &&
                      bcdata[i].channels == bcdataD[i].channels &&
                      bcdata[i].triggers == bcdataD[i].triggers &&
                      bcdata[i].ext_triggers == bcdataD[i].ext_triggers);

    if (!cmpBCData) {
      LOG(error) << "Mismatch in BC data " << i;
      bcdata[i].print();
      bcdataD[i].print();
    }
    BOOST_CHECK(cmpBCData);
  }

  LOG(info) << "Testing ChannelData: BOOST_CHECK(chandataD.size() " << chandataD.size() << " chandata.size()) " << chandata.size();
  BOOST_CHECK(chandataD.size() == chandata.size());

  for (size_t i = 0; i < chandata.size(); i++) {
    bool cmpChData = chandata[i].id == chandataD[i].id && chandata[i].data == chandataD[i].data;
    if (!cmpChData) {
      LOG(error) << "Mismatch in ChannelData " << i;
      chandata[i].print();
      chandataD[i].print();
    }
    BOOST_CHECK(cmpChData);
  }

  LOG(info) << "Testing OrbitData: BOOST_CHECK(pedsdataD.size() " << pedsdataD.size() << " pedsdata.size()) " << pedsdata.size();
  BOOST_CHECK(pedsdataD.size() == pedsdata.size());
  for (size_t i = 0; i < pedsdata.size(); i++) {
    bool cmpPdData = pedsdata[i].ir == pedsdataD[i].ir && pedsdata[i].data == pedsdataD[i].data && pedsdata[i].scaler == pedsdataD[i].scaler;
    if (!cmpPdData) {
      LOG(error) << "Mismatch in OrbitData " << i;
      pedsdata[i].print();
      pedsdataD[i].print();
    }
    BOOST_CHECK(cmpPdData);
  }
}
