// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test CPVCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DetectorsCommonDataFormats/NameConf.h"
#include "CPVReconstruction/CTFCoder.h"
#include "DataFormatsCPV/CTF.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <TSystem.h>
#include <cstring>

using namespace o2::cpv;

BOOST_AUTO_TEST_CASE(CTFTest)
{
  std::vector<TriggerRecord> triggers;
  std::vector<Cluster> clusters;
  TStopwatch sw;
  sw.Start();
  o2::InteractionRecord ir(0, 0);
  Cluster clu;
  for (int irof = 0; irof < 1000; irof++) {
    ir += 1 + gRandom->Integer(200);

    auto start = clusters.size();
    int n = 1 + gRandom->Poisson(100);
    for (int i = n; i--;) {
      char mult = gRandom->Integer(30);
      char mod = 1 + gRandom->Integer(3);
      char exMax = gRandom->Integer(3);
      float x = 72.3 * 2. * (gRandom->Rndm() - 0.5);
      float z = 63.3 * 2. * (gRandom->Rndm() - 0.5);
      float e = 254. * gRandom->Rndm();
      clusters.emplace_back(mult, mod, exMax, x, z, e);
    }
    triggers.emplace_back(ir, start, clusters.size() - start);
  }

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder;
    coder.encode(vec, triggers, clusters); // compress
  }
  sw.Stop();
  LOG(INFO) << "Compressed in " << sw.CpuTime() << " s";

  // writing
  {
    sw.Start();
    auto* ctfImage = o2::cpv::CTF::get(vec.data());
    TFile flOut("test_ctf_cpv.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "CPV");
    ctfTree.Write();
    sw.Stop();
    LOG(INFO) << "Wrote to tree in " << sw.CpuTime() << " s";
  }

  // reading
  vec.clear();
  LOG(INFO) << "Start reading from tree ";
  {
    sw.Start();
    TFile flIn("test_ctf_cpv.root");
    std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    BOOST_CHECK(tree);
    o2::cpv::CTF::readFromTree(vec, *(tree.get()), "CPV");
    sw.Stop();
    LOG(INFO) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<TriggerRecord> triggersD;
  std::vector<Cluster> clustersD;

  sw.Start();
  const auto ctfImage = o2::cpv::CTF::getImage(vec.data());
  {
    CTFCoder coder;
    coder.decode(ctfImage, triggersD, clustersD); // decompress
  }
  sw.Stop();
  LOG(INFO) << "Decompressed in " << sw.CpuTime() << " s";

  BOOST_CHECK(triggersD.size() == triggers.size());
  BOOST_CHECK(clustersD.size() == clusters.size());
  LOG(INFO) << " BOOST_CHECK triggersD.size() " << triggersD.size() << " triggers.size() " << triggers.size()
            << " BOOST_CHECK(clustersD.size() " << clustersD.size() << " clusters.size()) " << clusters.size();

  for (size_t i = 0; i < triggers.size(); i++) {
    const auto& dor = triggers[i];
    const auto& ddc = triggersD[i];
    LOG(DEBUG) << " Orig.TriggerRecord " << i << " " << dor.getBCData() << " " << dor.getFirstEntry() << " " << dor.getNumberOfObjects();
    LOG(DEBUG) << " Deco.TriggerRecord " << i << " " << ddc.getBCData() << " " << ddc.getFirstEntry() << " " << ddc.getNumberOfObjects();

    BOOST_CHECK(dor.getBCData() == ddc.getBCData());
    BOOST_CHECK(dor.getNumberOfObjects() == ddc.getNumberOfObjects());
    BOOST_CHECK(dor.getFirstEntry() == dor.getFirstEntry());
  }

  for (size_t i = 0; i < clusters.size(); i++) {
    const auto& cor = clusters[i];
    const auto& cdc = clustersD[i];
    BOOST_CHECK(cor.getMultiplicity() == cdc.getMultiplicity());
    BOOST_CHECK(cor.getModule() == cdc.getModule());
    BOOST_CHECK(TMath::Abs(cor.getEnergy() - cdc.getEnergy()) < 1.);
    float xCor, zCor, xCdc, zCdc;
    cor.getLocalPosition(xCor, zCor);
    cdc.getLocalPosition(xCdc, zCdc);
    BOOST_CHECK(TMath::Abs(xCor - xCdc) < 0.004);
    BOOST_CHECK(TMath::Abs(zCor - zCdc) < 0.004);
  }
}
