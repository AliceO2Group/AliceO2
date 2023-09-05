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

#define BOOST_TEST_MODULE Test TOFCTFIO
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#undef NDEBUG
#include <cassert>

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/dataset.hpp>
#include "DataFormatsTOF/CTF.h"
#include "TOFBase/Geo.h"
#include "TOFBase/Digit.h"
#include "CommonUtils/NameConf.h"
#include "TOFReconstruction/CTFCoder.h"
#include "Framework/Logger.h"
#include <TFile.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include <cstring>

using namespace o2::tof;
namespace boost_data = boost::unit_test::data;

inline std::vector<o2::ctf::ANSHeader> ANSVersions{o2::ctf::ANSVersionCompat, o2::ctf::ANSVersion1};

BOOST_DATA_TEST_CASE(CompressedClustersTest, boost_data::make(ANSVersions), ansVersion)
{

  std::vector<Digit> digits;
  std::vector<ReadoutWindowData> rows;
  std::vector<uint8_t> pattVec;

  TStopwatch sw;
  sw.Start();
  std::vector<int> row, col;
  for (int irof = 0; irof < 100; irof++) { // loop over row
    auto& rofr = rows.emplace_back();
    int orbit = irof / Geo::NWINDOW_IN_ORBIT;
    int bc = Geo::BC_IN_ORBIT / Geo::NWINDOW_IN_ORBIT * (irof % 3);
    rofr.SetOrbit(orbit);
    rofr.SetBC(bc);
    int ndig = gRandom->Poisson(50);

    rofr.setFirstEntry(digits.size());
    rofr.setNEntries(ndig);
    // fill empty pattern (to be changed)
    rofr.setFirstEntryDia(pattVec.size());
    rofr.setNEntriesDia(0);
    std::vector<int> istrip;
    for (int i = 0; i < ndig; i++) {
      istrip.emplace_back(gRandom->Integer(Geo::NSTRIPS));
    }
    std::sort(istrip.begin(), istrip.end());

    for (int i = 0; i < ndig; i++) {
      int ch = istrip[i] * Geo::NPADS + gRandom->Integer(Geo::NPADS);
      uint16_t TDC = gRandom->Integer(1024);
      uint16_t TOT = gRandom->Integer(2048);
      uint64_t BC = Geo::BC_IN_ORBIT * orbit + bc + gRandom->Integer(Geo::BC_IN_ORBIT / Geo::NWINDOW_IN_ORBIT);

      digits.emplace_back(ch, TDC, TOT, BC);
    }

    std::sort(digits.begin(), digits.end(),
              [](const Digit& a, const Digit& b) {
                int strip1 = a.getChannel() / Geo::NPADS, strip2 = b.getChannel() / Geo::NPADS;
                if (strip1 == strip2) {
                  if (a.getBC() == b.getBC()) {
                    return a.getTDC() < b.getTDC();
                  }
                  return a.getBC() < b.getBC();
                }
                return strip1 < strip2;
              });

    //    for (int i = 0; i < ndig; i++)
    //        LOG(info) << "ROW = " << irof << " - Strip = " << digits[i].getChannel() / Geo::NPADS << " - BC = " << digits[i].getBC() << " - TDC = " << digits[i].getTDC();
  }
  sw.Stop();
  LOG(info) << "Generated " << digits.size() << " in " << rows.size() << " ROFs in " << sw.CpuTime() << " s";

  sw.Start();
  std::vector<o2::ctf::BufferType> vec;
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Encoder);
    coder.setANSVersion(ansVersion);
    coder.encode(vec, rows, digits, pattVec); // compress
  }
  sw.Stop();
  LOG(info) << "Compressed in " << sw.CpuTime() << " s";

  // writing
  {
    sw.Start();
    TFile flOut("test_ctf_tof.root", "recreate");
    TTree ctfTree(std::string(o2::base::NameConf::CTFTREENAME).c_str(), "O2 CTF tree");
    auto* ctfImage = CTF::get(vec.data());
    ctfImage->print();
    ctfImage->appendToTree(ctfTree, "TOF");
    ctfTree.Write();
    sw.Stop();
    LOG(info) << "Wrote to tree in " << sw.CpuTime() << " s";
  }

  // reading
  vec.clear();
  {
    sw.Start();
    TFile flIn("test_ctf_tof.root");
    std::unique_ptr<TTree> tree((TTree*)flIn.Get(std::string(o2::base::NameConf::CTFTREENAME).c_str()));
    BOOST_CHECK(tree);
    CTF::readFromTree(vec, *(tree.get()), "TOF");
    sw.Stop();
    LOG(info) << "Read back from tree in " << sw.CpuTime() << " s";
  }

  std::vector<Digit> digitsD;
  std::vector<ReadoutWindowData> rowsD;
  std::vector<uint8_t> pattVecD;
  sw.Start();
  const auto ctfImage = CTF::getImage(vec.data());
  {
    CTFCoder coder(o2::ctf::CTFCoderBase::OpType::Decoder);
    coder.decode(ctfImage, rowsD, digitsD, pattVecD); // decompress
  }
  sw.Stop();
  LOG(info) << "Decompressed in " << sw.CpuTime() << " s";

  //
  // simple checks
  BOOST_CHECK(rows.size() == rowsD.size());
  BOOST_CHECK(digits.size() == digitsD.size());
  BOOST_CHECK(pattVec.size() == pattVecD.size());

  // more sophisticated checks

  // checks on patterns
  int npatt = ctfImage.getHeader().nPatternBytes;
  for (int i = 0; i < npatt; i += 100) {
    BOOST_CHECK(pattVecD[i] == pattVec[i]);
  }
}
