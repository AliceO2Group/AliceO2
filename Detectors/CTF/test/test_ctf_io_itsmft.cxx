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

#define BOOST_TEST_MODULE Test ITSMFTCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/CTF.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonUtils/NameConf.h"
#include "ITSMFTReconstruction/CTFCoder.h"
#include "ITSMFTReconstruction/LookUp.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <cstring>

using namespace o2::itsmft;

BOOST_AUTO_TEST_CASE(CompressedClustersTest)
{

  std::vector<ROFRecord> rofRecVec;
  std::vector<CompClusterExt> cclusVec;
  std::vector<unsigned char> pattVec;
  LookUp pattIdConverter;
  TStopwatch sw;
  sw.Start();
  std::vector<int> row, col;
  for (int irof = 0; irof < 100; irof++) {
    auto& rofr = rofRecVec.emplace_back();
    rofr.getBCData().orbit = irof / 10;
    rofr.getBCData().bc = irof % 10;
    int nChips = 5 * irof;
    int chipID = irof / 2;
    rofr.setFirstEntry(cclusVec.size());
    for (int i = 0; i < nChips; i++) {
      int nhits = gRandom->Poisson(50);
      row.resize(nhits);
      col.resize(nhits);
      for (int i = 0; i < nhits; i++) {
        row[i] = gRandom->Integer(512);
        col[i] = gRandom->Integer(1024);
      }
      std::sort(col.begin(), col.end());
      for (int i = 0; i < nhits; i++) {
        auto& cl = cclusVec.emplace_back(row[i], col[i], gRandom->Integer(1000), chipID);
        if (cl.getPatternID() > 900) {
          int nbpatt = 1 + gRandom->Poisson(3.);
          for (int i = nbpatt; i--;) {
            pattVec.push_back(char(gRandom->Integer(256)));
          }
        }
      }
      chipID += 1 + gRandom->Poisson(10);
    }
    rofr.setNEntries(int(cclusVec.size()) - rofr.getFirstEntry());
  }
  sw.Stop();
  LOG(info) << "Generated " << cclusVec.size() << " in " << rofRecVec.size() << " ROFs in " << sw.CpuTime() << " s";

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Encoder, o2::detectors::DetID::ITS);
    coder.encode(vec, rofRecVec, cclusVec, pattVec, pattIdConverter, 0); // compress
  }
  sw.Stop();
  LOG(info) << "Compressed in " << sw.CpuTime() << " s";

  // writing
  {
    sw.Start();
    TFile flOut("test_ctf_itsmft.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    auto* ctfImage = o2::itsmft::CTF::get(vec.data());
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "ITS");
    ctfTree.Write();
    sw.Stop();
    LOG(info) << "Wrote to tree in " << sw.CpuTime() << " s";
  }

  // reading
  vec.clear();
  {
    sw.Start();
    TFile flIn("test_ctf_itsmft.root");
    std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    BOOST_CHECK(tree);
    o2::itsmft::CTF::readFromTree(vec, *(tree.get()), "ITS");
    sw.Stop();
    LOG(info) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<ROFRecord> rofRecVecD;
  std::vector<CompClusterExt> cclusVecD;
  std::vector<unsigned char> pattVecD;
  LookUp clPattLookup;
  sw.Start();
  const auto ctfImage = o2::itsmft::CTF::getImage(vec.data());
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Decoder, o2::detectors::DetID::ITS);
    coder.decode(ctfImage, rofRecVecD, cclusVecD, pattVecD, nullptr, clPattLookup); // decompress
  }
  sw.Stop();
  LOG(info) << "Decompressed in " << sw.CpuTime() << " s";

  //
  // check
  BOOST_CHECK(rofRecVecD.size() == rofRecVec.size());
  BOOST_CHECK(cclusVecD.size() == cclusVec.size());
  BOOST_CHECK(pattVecD.size() == pattVec.size());
  int di = rofRecVec.size() / 10 ? rofRecVec.size() / 10 : 1;
  for (int i = 0; i < int(rofRecVec.size()); i += di) {
    BOOST_CHECK(rofRecVecD[i].getBCData() == rofRecVec[i].getBCData());
    BOOST_CHECK(rofRecVecD[i].getFirstEntry() == rofRecVec[i].getFirstEntry());
    BOOST_CHECK(rofRecVecD[i].getNEntries() == rofRecVec[i].getNEntries());
    //
    int ncl = rofRecVec[i].getNEntries();
    int firstCl = rofRecVec[i].getFirstEntry();
    for (int j = 0; j < ncl; j += 10) {
      auto j1 = j + firstCl;
      BOOST_CHECK(cclusVecD[j1].getChipID() == cclusVec[j1].getChipID());
      BOOST_CHECK(cclusVecD[j1].getRow() == cclusVec[j1].getRow());
      BOOST_CHECK(cclusVecD[j1].getCol() == cclusVec[j1].getCol());
    }
  }
  //
  int npatt = ctfImage.getHeader().nPatternBytes;
  for (int i = 0; i < npatt; i += 100) {
    BOOST_CHECK(pattVecD[i] == pattVec[i]);
  }
}
