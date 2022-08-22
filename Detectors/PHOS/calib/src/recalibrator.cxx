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

/// @file   recalibrator.cxx
/// @author Dmitri Peresunko
/// @since  2022-07-14
/// @brief  executable to recalibrate and recalculate calibration hitograms
#include "PHOSBase/Geometry.h"
#include "PHOSBase/PHOSSimParams.h"
#include "PHOSCalibWorkflow/ETCalibHistos.h"
#include "PHOSCalibWorkflow/PHOSEnergyCalibrator.h"
#include "DataFormatsPHOS/CalibParams.h"
#include "DataFormatsPHOS/BadChannelsMap.h"
#include "DataFormatsPHOS/Cluster.h"
#include <boost/program_options.hpp>
#include <boost/timer/progress_display.hpp>
#include "TChain.h"
#include "TH1F.h"
#include "TH2F.h"

namespace bpo = boost::program_options;

void evalAll(o2::phos::Cluster* clu, std::vector<o2::phos::CluElement>& cluel);

int main(int argc, const char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       " <cmds/options>\n"
                                       "  Tool will recalibrate phos calib digits and produce inv. mass histos\n"
                                       "Commands / Options");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("input,i", bpo::value<std::string>()->default_value("list"), "List of input *.root files");
    add_option("calib,c", bpo::value<std::string>()->default_value("Calib.root"), "Current calibration");
    add_option("badmap,b", bpo::value<std::string>()->default_value("BadMap.root"), "Bad channels map or none");
    add_option("ptmin,p", bpo::value<float>()->default_value(0.5), "Min pT for inv mass analysis");
    opt_all.add(opt_general).add(opt_hidden);
    bpo::store(bpo::command_line_parser(argc, argv).options(opt_all).positional(opt_pos).run(), vm);

    if (vm.count("help") || argc == 1) {
      std::cout << opt_general << std::endl;
      exit(0);
    }

  } catch (bpo::error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl
              << std::endl;
    std::cerr << opt_general << std::endl;
    exit(1);
  } catch (std::exception& e) {
    std::cerr << e.what() << ", application will now exit" << std::endl;
    exit(2);
  }

  std::string input = "list";
  if (vm.count("input")) {
    input = vm["input"].as<std::string>();
    if (input.empty()) {
      std::cerr << "Please provide input file list with option --input-files list" << std::endl;
      exit(1);
    }
  }

  std::string calibFile = "Calib.root";
  if (vm.count("calib")) {
    calibFile = vm["calib"].as<std::string>();
  }

  std::string badmapFile;
  if (vm.count("badmap")) {
    badmapFile = vm["badmap"].as<std::string>();
  }
  float ptMin = 0.5;
  if (vm.count("ptmin")) {
    ptMin = vm["ptmin"].as<float>();
  }

  // Read current calibration
  o2::phos::CalibParams* calibParams = new o2::phos::CalibParams(1);

  o2::phos::BadChannelsMap* badmap = nullptr;
  if (!badmapFile.empty() && badmapFile != "none") {
    TFile* fBadMap = TFile::Open(badmapFile.c_str());
    if (fBadMap->IsOpen()) {
      badmap = (o2::phos::BadChannelsMap*)fBadMap->Get("BadMap");
    } else {
      badmap = new o2::phos::BadChannelsMap(); // default/empty bad map
      std::cout << "Using empty bad map" << std::endl;
    }
  } else {
    badmap = new o2::phos::BadChannelsMap(); // default/empty bad map
    std::cout << "Using empty bad map" << std::endl;
  }

  // Scan data
  TChain* chain = new TChain("phosCalibDig");
  std::ifstream fin(input);
  if (!fin.is_open()) {
    std::cerr << "can not open file " << input << std::endl;
    exit(1);
  }
  while (!fin.eof()) {
    std::string fname;
    fin >> fname;
    if (!fname.empty()) {
      chain->AddFile(fname.c_str());
    }
  }
  fin.close();

  o2::phos::Geometry::GetInstance("Run3");
  // Histogram init
  static constexpr int nChannels = 14336 - 1793; // 4 full modules -1/2
  static constexpr int offset = 1793;            // 1/2 full module
  static constexpr int nMass = 150.;
  static constexpr float massMax = 0.3;
  static constexpr int npt = 200;
  static constexpr float ptMax = 20;
  TH1F* hNev = new TH1F("hNev", "Number of events", 2, 0., 2.);
  TH2F* hNonLinRe = new TH2F("hNonLinRe", "Nonlinearity", npt, 0, ptMax, nMass, 0., massMax);
  TH2F* hNonLinMi = new TH2F("hNonLinMi", "Nonlinearity", npt, 0, ptMax, nMass, 0., massMax);
  TH2F* hMassPerCellRe = new TH2F("hMassPerCellRe", "MinvRe", nChannels, offset, nChannels + offset, nMass, 0., massMax);
  TH2F* hMassPerCellMi = new TH2F("hMassPerCellMi", "MinvRe", nChannels, offset, nChannels + offset, nMass, 0., massMax);

  std::vector<uint32_t>* digits = nullptr;
  chain->SetBranchAddress("PHOSCalib", &digits);

  o2::phos::RingBuffer buffer;

  o2::phos::EventHeader h = {0};
  o2::phos::CalibDigit cd = {0};
  std::vector<o2::phos::CluElement> cluelements;
  std::vector<o2::phos::Cluster> clusters;
  std::list<o2::phos::Cluster> mixedClu;
  o2::phos::Cluster* clu = nullptr;
  TVector3 globaPos;

  boost::timer::progress_display progress(100);
  for (int i = 0; i < chain->GetEntries(); i++) {
    // Print progress
    if (i % (chain->GetEntries() / 100) == 0) {
      ++progress;
    }
    chain->GetEvent(i);
    auto d = digits->begin();
    int bc = -1, orbit = -1, currentCluIndex = -1; // not defined yet
    while (d != digits->end()) {
      h.mDataWord = *d;
      if (h.mMarker == 16383) { // new event marker
        if (bc > -1) {          // process previous event

          buffer.startNewEvent(); // mark stored clusters to be used for Mixing
          hNev->Fill(1);
          for (o2::phos::Cluster& c : clusters) {
            short absId;
            float x, z;
            c.getLocalPosition(x, z);
            o2::phos::Geometry::GetInstance()->relPosToAbsId(c.module(), x, z, absId);
            o2::phos::Geometry::GetInstance()->local2Global(c.module(), x, z, globaPos);
            float e = c.getEnergy();
            double sc = e / globaPos.Mag();
            TLorentzVector v(sc * globaPos.X(), sc * globaPos.Y(), sc * globaPos.Z(), e);
            bool isGood = true; // badmap->isChannelGood(absId);
            for (short ip = buffer.size(); ip--;) {
              const TLorentzVector& vp = buffer.getEntry(ip);
              TLorentzVector sum = v + vp;
              if (buffer.isCurrentEvent(ip)) { // same (real) event
                if (isGood) {
                  hNonLinRe->Fill(e, sum.M());
                }
                if (sum.Pt() > ptMin) {
                  hMassPerCellRe->Fill(absId, sum.M());
                }
              } else { // Mixed
                if (isGood) {
                  hNonLinMi->Fill(e, sum.M());
                }
                if (sum.Pt() > ptMin) {
                  hMassPerCellMi->Fill(absId, sum.M());
                }
              }
            }
            // Add to list ot partners only if cluster is good
            if (isGood) {
              buffer.addEntry(v);
            }
          }
          clusters.clear();
          currentCluIndex = -1;
        }
        bc = h.mBC;
        d++;
        if (d == digits->end()) {
          std::cout << "No orbit number, exit" << std::endl;
          exit(1);
        }
        orbit = *d;
        d++;
        if (d == digits->end()) { // no digits in event
          break;
        }
      } else { // normal digit
        if (bc < 0) {
          std::cout << "Corrupted data: no header" << std::endl;
          exit(1);
        }
        // read new digit
        cd.mDataWord = *d;
        d++;
        short absId = cd.mAddress;
        short adcCounts = cd.mAdcAmp;
        float e = calibParams->getGain(absId) * adcCounts;
        bool isHG = cd.mHgLg;
        float x = 0., z = 0.;
        o2::phos::Geometry::absIdToRelPosInModule(absId, x, z);
        int cluIndex = cd.mCluster;
        if (cluIndex != currentCluIndex) {
          if (currentCluIndex >= 0) {
            clu->setLastCluEl(cluelements.size());
            evalAll(clu, cluelements);
          }
          // start new cluster
          clusters.emplace_back();
          clu = &(clusters.back());
          clu->setFirstCluEl(cluelements.size());
          currentCluIndex = cluIndex;
        }
        cluelements.emplace_back(absId, isHG, e, 0., x, z, -1, 0.);
      }
    } // digits loop
  }
  TFile fout("histos.root", "recreate");
  hNev->Write();
  hNonLinRe->Write();
  hNonLinMi->Write();
  hMassPerCellRe->Write();
  hMassPerCellMi->Write();
  fout.Close();
}
//____________________________________________________________________________
void evalAll(o2::phos::Cluster* clu, std::vector<o2::phos::CluElement>& cluel)
{
  // position, energy, coreEnergy, dispersion, time,

  // Calculates the center of gravity in the local PHOS-module coordinates
  // Note that correction for non-perpendicular incidence will be applied later
  // when vertex will be known.
  float fullEnergy = 0.;
  uint32_t iFirst = clu->getFirstCluEl(), iLast = clu->getLastCluEl();
  clu->setModule(o2::phos::Geometry::absIdToModule(cluel[iFirst].absId));
  float eMin = o2::phos::PHOSSimParams::Instance().mDigitMinEnergy;
  for (uint32_t i = iFirst; i < iLast; i++) {
    float ei = cluel[i].energy;
    if (ei < eMin) {
      continue;
    }
    fullEnergy += ei;
  }
  clu->setEnergy(fullEnergy);
  if (fullEnergy <= 0) {
    return;
  }
  // Calculate time as time in the digit with maximal energy
  clu->setTime(0.);

  float localPosX = 0., localPosZ = 0.;
  float wtot = 0.;
  float invE = 1. / fullEnergy;
  for (uint32_t i = iFirst; i < iLast; i++) {
    o2::phos::CluElement& ce = cluel[i];
    if (ce.energy < eMin) {
      continue;
    }
    float w = std::max(float(0.), o2::phos::PHOSSimParams::Instance().mLogWeight + std::log(ce.energy * invE));
    localPosX += ce.localX * w;
    localPosZ += ce.localZ * w;
    wtot += w;
  }
  if (wtot > 0) {
    wtot = 1. / wtot;
    localPosX *= wtot;
    localPosZ *= wtot;
  }
  clu->setLocalPosition(localPosX, localPosZ);
}