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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TChain.h>
#include <TFile.h>
#include <TH1F.h>
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/LinkData.h"
#include <memory>
#endif

//Parse the trackletraw output of a trap simulation
//Display on the screen and indent for half chamber, mcmheader and mcmtracklet.
//Create a raw dump as well of the incoming data.

using namespace o2::trd;

void ParseTrapRawOutput(std::string path = "./", std::string inputTracklets = "trdtrapraw.root")
{
  TChain trdtrapraw("o2sim");
  trdtrapraw.AddFile((path + inputTracklets).c_str());

  std::vector<o2::trd::LinkRecord> links;
  std::vector<o2::trd::LinkRecord>* linksptr = &links;

  std::vector<uint32_t> trapraw;
  std::vector<uint32_t>* traprawptr = &trapraw;
  trdtrapraw.SetBranchAddress("TrapRaw", &traprawptr);
  trdtrapraw.SetBranchAddress("TrapLinkRecord", &linksptr);
  trdtrapraw.GetEntry(0);

  ofstream outfile("rawdata", std::ios::out | std::ofstream::binary);
  outfile.write((char*)(&trapraw[0]), sizeof(trapraw[0]) * trapraw.size());
  outfile.close();
  uint64_t mcmheadcount = 0;
  uint64_t halfchamberheadcount = 0;
  uint64_t traprawtrackletount = 0;
  //at each linkrecord data we should have a halfchamberheader;
  // with in the range specified by the linkrecord we have a structure of :
  // mcmheader, traprawtracklet[1-3], mcmheader, traprawtracklet[1-3], etc. etc.
  for (auto& link : links) {
    o2::trd::TrackletHCHeader halfchamber;
    halfchamber.word = link.getLinkId();
    std::cout << "in link with HCID of " << halfchamber;
    for (int i = link.getFirstEntry(); i < link.getFirstEntry() + link.getNumberOfObjects(); i++) {
      //read TrackletMCMHeader
      //read 1 to 3 TrackletMCMDatas
      if (((trapraw[i]) & 0x1) == 1 && (trapraw[i] & 0x80000000) != 0) {
        TrackletMCMHeader mcm;
        mcm.word = trapraw[i];
        std::cout << "\t\t" << mcm;
        mcmheadcount++;
      } else if (((trapraw[i]) & 0x1) == 0) {
        //tracklet word
        //
        TrackletMCMData tracklet;
        tracklet.word = trapraw[i];
        std::cout << "\t\t\t\tTracklet : 0x" << std::hex << tracklet.word << std::dec << std::endl;
        traprawtrackletount++;
      } else
        std::cout << "most sig bit is not binary ?? 0x" << std::hex << trapraw[i] << std::dec << std::endl;
    }
  }
  std::cout << "counts " << halfchamberheadcount << "::" << mcmheadcount << "::" << traprawtrackletount << std::endl;
}
