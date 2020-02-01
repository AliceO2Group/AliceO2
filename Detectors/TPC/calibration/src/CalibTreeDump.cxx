// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <fmt/format.h>
#include <cstdlib>
#include <string_view>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>

#include "TFile.h"
#include "TTree.h"
#include "TMath.h"

#include "MathUtils/MathBase.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CRU.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/Sector.h"

#include "TPCCalibration/CalibTreeDump.h"

using o2::math_utils::math_base::median;

using namespace o2::tpc;

//______________________________________________________________________________
void CalibTreeDump::dumpToFile(const std::string filename)
{
  // ===| open file and crate tree |============================================
  std::unique_ptr<TFile> file(TFile::Open(filename.c_str(), "recreate"));
  auto tree = new TTree("calibTree", "Calibration data object tree");

  // ===| add default mappings |================================================
  addDefaultMapping(tree);

  // ===| add FEE mapping |=====================================================
  if (mAddFEEInfo) {
    addFEEMapping(tree);
  }

  // ===| fill calDet objects |=================================================
  addCalDetObjects(tree);

  // ===| default aliases |=====================================================
  setDefaultAliases(tree);

  file->Write();
}

//______________________________________________________________________________
void CalibTreeDump::addDefaultMapping(TTree* tree)
{
  // loop over ROCs
  //

  // ===| mapper |==============================================================
  const auto& mapper = Mapper::instance();

  // ===| default mapping objects |=============================================
  uint16_t rocNumber = 0;
  // positions
  std::vector<float> gx;
  std::vector<float> gy;
  std::vector<float> lx;
  std::vector<float> ly;
  // row and pad
  std::vector<unsigned char> row;
  std::vector<unsigned char> pad;
  std::vector<short> cpad;

  // ===| add branches with default mappings |==================================
  tree->Branch("roc", &rocNumber);
  tree->Branch("gx", &gx);
  tree->Branch("gy", &gy);
  tree->Branch("lx", &lx);
  tree->Branch("ly", &ly);
  tree->Branch("row", &row);
  tree->Branch("pad", &pad);
  tree->Branch("cpad", &cpad);

  // ===| loop over readout chambers |==========================================
  for (ROC roc; !roc.looped(); ++roc) {
    rocNumber = roc;

    // ===| clear position vectors |============================================
    gx.clear();
    gy.clear();
    lx.clear();
    ly.clear();
    row.clear();
    pad.clear();
    cpad.clear();

    // ===| loop over pad rows |================================================
    const int numberOfRows = mapper.getNumberOfRowsROC(roc);
    for (int irow = 0; irow < numberOfRows; ++irow) {

      // ===| loop over pads in row |===========================================
      const int numberOfPadsInRow = mapper.getNumberOfPadsInRowROC(roc, irow);
      for (int ipad = 0; ipad < numberOfPadsInRow; ++ipad) {
        const PadROCPos padROCPos(rocNumber, irow, ipad);
        const PadPos padPos = mapper.getGlobalPadPos(padROCPos); // pad and row in sector
        const PadCentre& localPadXY = mapper.getPadCentre(padPos);
        const LocalPosition2D globalPadXY = mapper.getPadCentre(padROCPos);

        gx.emplace_back(globalPadXY.X());
        gy.emplace_back(globalPadXY.Y());
        lx.emplace_back(localPadXY.X());
        ly.emplace_back(localPadXY.Y());

        row.emplace_back(irow);
        pad.emplace_back(ipad);
        cpad.emplace_back(ipad - numberOfPadsInRow / 2);
      }
    }

    tree->Fill();
  }
}

//______________________________________________________________________________
void CalibTreeDump::addFEEMapping(TTree* tree)
{
  // loop over ROCs
  //

  // ===| mapper |==============================================================
  const auto& mapper = Mapper::instance();

  readTraceLengths();

  // ===| default mapping objects |=============================================
  // FEC
  std::vector<unsigned char> fecInSector;
  std::vector<unsigned char> sampaOnFEC;
  std::vector<unsigned char> channelOnSampa;
  // trace length
  std::vector<float>* traceLength = nullptr;

  // ===| add branches with default mappings |==================================
  auto brFecInSector = tree->Branch("fecInSector", &fecInSector);
  auto brSampaOnFEC = tree->Branch("sampaOnFEC", &sampaOnFEC);
  auto brChannelOnSampa = tree->Branch("channelOnSampa", &channelOnSampa);
  auto brTraceLength = tree->Branch("traceLength", &traceLength);

  // ===| loop over readout chambers |==========================================
  for (ROC roc; !roc.looped(); ++roc) {
    int rocNumber = roc;
    //tree->GetEntry(rocNumber);
    traceLength = ((roc.rocType() == RocType::IROC) ? &mTraceLengthIROC : &mTraceLengthOROC);

    // ===| clear position vectors |============================================
    fecInSector.clear();
    sampaOnFEC.clear();
    channelOnSampa.clear();

    const int rowOffset = (roc.rocType() == RocType::OROC) ? mapper.getPadsInIROC() : 0;
    // ===| loop over pad rows |================================================
    const int numberOfRows = mapper.getNumberOfRowsROC(roc);
    for (int irow = 0; irow < numberOfRows; ++irow) {

      // ===| loop over pads in row |===========================================
      const int numberOfPadsInRow = mapper.getNumberOfPadsInRowROC(roc, irow);
      for (int ipad = 0; ipad < numberOfPadsInRow; ++ipad) {
        const PadROCPos padROCPos(rocNumber, irow, ipad);
        const auto& fecInfo = mapper.getFECInfo(padROCPos);
        const auto globalPadNumber = mapper.globalPadNumber(fecInfo);

        const CRU cru(mapper.getCRU(roc.getSector(), globalPadNumber));

        fecInSector.emplace_back(fecInfo.getIndex());
        sampaOnFEC.emplace_back(fecInfo.getSampaChip());
        channelOnSampa.emplace_back(fecInfo.getSampaChannel());
      }
    }
    brFecInSector->Fill();
    brSampaOnFEC->Fill();
    brChannelOnSampa->Fill();
    brTraceLength->Fill();
  }
}

//______________________________________________________________________________
void CalibTreeDump::addCalDetObjects(TTree* tree)
{

  int iter = 0;
  for (auto& calDet : mCalDetObjects) {
    // ===| branch names |===
    std::string name = calDet.getName();

    if (name == "PadCalibrationObject" || name.size() == 0) {
      name = fmt::format("calDet_{%02d}", iter);
    }

    std::string meanName = fmt::format("{}_mean", name);
    std::string stdDevName = fmt::format("{}_stdDev", name);
    std::string medianName = fmt::format("{}_median", name);

    // ===| branch variables |===
    std::vector<float>* data = nullptr;
    float mean{};
    float stdDev{};
    float median{};

    // ===| branch definitions |===
    TBranch* brMean = tree->Branch(meanName.data(), &mean);
    TBranch* brStdDev = tree->Branch(stdDevName.data(), &stdDev);
    TBranch* brMedian = tree->Branch(medianName.data(), &median);
    TBranch* brData = tree->Branch(name.data(), &data);

    // ===| loop over ROCs and fill |===
    for (auto& calArray : calDet.getData()) {
      // ---| set data |---
      data = &calArray.getData();

      // ---| statistics |---
      mean = TMath::Mean(data->begin(), data->end());
      stdDev = TMath::StdDev(data->begin(), data->end());
      median = TMath::Median(data->size(), data->data());

      // ---| filling |---
      brData->Fill();
      brMean->Fill();
      brStdDev->Fill();
      brMedian->Fill();
    }
  }
}

//______________________________________________________________________________
void CalibTreeDump::readTraceLengths(std::string_view mappingDir)
{
  std::string inputDir = mappingDir.data();
  if (!inputDir.size()) {
    const char* aliceO2env = std::getenv("O2_ROOT");
    if (aliceO2env)
      inputDir = aliceO2env;
    inputDir += "/share/Detectors/TPC/files";
  }

  mTraceLengthIROC.reserve(Mapper::getPadsInIROC());
  mTraceLengthOROC.reserve(Mapper::getPadsInOROC());
  setTraceLengths(inputDir + "/LENGTH-IROC.txt", mTraceLengthIROC);
  setTraceLengths(inputDir + "/LENGTH-OROC1.txt", mTraceLengthOROC);
  setTraceLengths(inputDir + "/LENGTH-OROC2.txt", mTraceLengthOROC);
  setTraceLengths(inputDir + "/LENGTH-OROC3.txt", mTraceLengthOROC);

  assert(mTraceLengthIROC.size() == Mapper::getPadsInIROC());
  assert(mTraceLengthOROC.size() == Mapper::getPadsInOROC());
}

//______________________________________________________________________________
void CalibTreeDump::setTraceLengths(std::string_view inputFile, std::vector<float>& length)
{

  std::ifstream infile(inputFile.data(), std::ifstream::in);
  if (!infile.is_open()) {
    std::cout << "could not open file " << inputFile.data() << "\n";
    exit(1);
  }

  // e.g. IROC file
  // Col 0 -> INDEX (0 - 5279)
  // Col 1 -> PADROW (0 - 62)
  // Col 2 -> PAD (0 - (Np-1))
  // Col 3 -> Connector (1 - 132)
  // Col 4 -> Pin (1 - 40)
  // Col 5 -> Trace length (mm)
  // Col 6 -> Number of vias

  unsigned int index{};
  unsigned int padrow{};
  unsigned int pad{};
  unsigned int connector{};
  unsigned int pin{};
  float traceLength{};
  unsigned int numberOfVias{};

  std::string line;
  while (std::getline(infile, line)) {
    std::stringstream streamLine(line);
    streamLine >> index >> padrow >> pad >> connector >> pin >> traceLength >> numberOfVias;

    traceLength /= 10.f;

    length.emplace_back(traceLength);
  }
}

//______________________________________________________________________________
void CalibTreeDump::setDefaultAliases(TTree* tree)
{
  tree->SetAlias("sector", "roc%36");
  tree->SetAlias("rowInSector", "row + (roc>35)*63");
  tree->SetAlias("padWidth", "0.4 + (roc > 35) * 0.2");
  tree->SetAlias("padHeight", "0.75 + (rowInSector > 62) * 0.25 + (rowInSector > 96) * 0.2 + (rowInSector > 126) * 0.3");
  tree->SetAlias("padArea", "padWidth * padHeight");

  tree->SetAlias("cruInSector", "(rowInSector >= 17) + (rowInSector >= 32) + (rowInSector >= 48) + (rowInSector >= 63) + (rowInSector >= 81) + (rowInSector >= 97) + (rowInSector >= 113) + (rowInSector >= 127) + (rowInSector >= 140)");
  tree->SetAlias("cruID", "cruInSector + sector*10");
  tree->SetAlias("region", "cruInSector");
  tree->SetAlias("partition", "int(cruInSector / 2)");

  tree->SetAlias("IROC", "roc < 36");
  tree->SetAlias("OROC", "roc >= 36");
  tree->SetAlias("OROC1", "partition == 2");
  tree->SetAlias("OROC2", "partition == 3");
  tree->SetAlias("OROC3", "partition == 4");

  tree->SetAlias("A_Side", "sector < 36");
  tree->SetAlias("C_Side", "sector >= 36");

  if (mAddFEEInfo) {
    tree->SetAlias("fecID", "fecInSector + sector * 91");
    tree->SetAlias("sampaInSector", "sampaOnFEC + 5 * fecInSector");
    tree->SetAlias("channelOnFEC", "channelOnSampa + 32 * sampaOnFEC");
    tree->SetAlias("channelInSector", "channelOnFEC + 160 * fecInSector");
  }
}
