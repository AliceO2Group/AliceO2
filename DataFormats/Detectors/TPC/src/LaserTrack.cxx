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

/// \file LaserTrack.cxx
/// \brief Laser track parameters
/// \author Jens Wiechula, jens.wiechula@ikf.uni-frankfurt.de

#include <memory>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <fairlogger/Logger.h>

#include "TFile.h"
#include "TTree.h"

#include "DataFormatsTPC/LaserTrack.h"

using namespace o2::tpc;

void LaserTrackContainer::loadTracksFromFile()
{
  int id;
  float x, alpha, p0, p1, p2, p3, p4;

  const std::string o2Root(std::getenv("O2_ROOT"));
  const std::string file = o2Root + "/share/Detectors/TPC/files/laserTrackData.txt";

  std::string line;
  std::ifstream infile(file, std::ifstream::in);

  if (!infile.is_open()) {
    LOG(error) << "Could not open laser track file " << file;
    return;
  }

  while (std::getline(infile, line)) {
    std::stringstream streamLine(line);
    streamLine >> id >> x >> alpha >> p0 >> p1 >> p2 >> p3 >> p4;

    //printf("%3d: %f %f %f %f %f %f %f \n", id, x, alpha, p0, p1, p2, p3, p4);
    mLaserTracks[id] = LaserTrack(id, x, alpha, {p0, p1, p2, p3, p4});
  }
}

void LaserTrackContainer::dumpToTree(const std::string_view fileName)
{
  LaserTrackContainer c;
  c.loadTracksFromFile();
  const auto& tracks = c.getLaserTracks();
  std::vector<LaserTrack> vtracks;

  for (const auto& track : tracks) {
    vtracks.emplace_back(track);
  }

  std::unique_ptr<TFile> fout(TFile::Open(fileName.data(), "recreate"));
  TTree t("laserTracks", "Laser Tracks");
  t.Branch("tracks", &vtracks);
  t.Fill();
  fout->Write();
  fout->Close();
}
