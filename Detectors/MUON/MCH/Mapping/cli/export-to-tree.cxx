// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHMappingInterface/Segmentation.h"
#include <iostream>
#include <fmt/format.h>
#include <boost/program_options.hpp>
#include <TTree.h>
#include <TFile.h>
#include <stdexcept>

namespace po = boost::program_options;
using namespace o2::mch::mapping;

#ifdef RUN2
constexpr bool run2{true};
#else
constexpr bool run2{false};
#endif

constexpr int MaxNofPadsPerDE{28672};
struct DePads {
  int nDePad;
  int deid;
  int dsid[MaxNofPadsPerDE];
  int dsch[MaxNofPadsPerDE];
  float x[MaxNofPadsPerDE];
  float y[MaxNofPadsPerDE];
  float dx[MaxNofPadsPerDE];
  float dy[MaxNofPadsPerDE];
};

void createPadBranches(TTree& tree, DePads& depads)
{
  tree.Branch("nDePad", &depads.nDePad, "nDePad/I")
    ->SetTitle("Number of pads in detection element");
  tree.Branch("deid", &depads.deid, "deid/I")
    ->SetTitle("Detection element id");
  tree.Branch(fmt::format("DePad_{}", run2 ? "manu" : "dsid").c_str(),
              &depads.dsid,
              fmt::format("DePad_{}[nDePad]/I", run2 ? "manu" : "dsid").c_str())
    ->SetTitle("FEE id of n-th pad of this detection element (cm)");
  tree.Branch(fmt::format("DePad_{}", run2 ? "ch" : "dsch").c_str(),
              &depads.dsch,
              fmt::format("DePad_{}[nDePad]/I", run2 ? "ch" : "dsch").c_str())
    ->SetTitle("FEE channel of n-th pad of this detection element (cm)");
  tree.Branch("DePad_x", &depads.x, "DePad_x[nDePad]/F")
    ->SetTitle("x position of n-th pad of this detection element (cm)");
  tree.Branch("DePad_y", &depads.y, "DePad_y[nDePad]/F")
    ->SetTitle("y position of n-th pad of this detection element (cm)");
  tree.Branch("DePad_dx", &depads.dx, "DePad_dx[nDePad]/F")
    ->SetTitle("half size in x direction of n-th pad of this detection element (cm)");
  tree.Branch("DePad_dy", &depads.dy, "DePad_dy[nDePad]/F")
    ->SetTitle("half size in y direction of n-th pad of this detection element (cm)");
}

/**
 * This small program creates a Root TTree with MCH mapping.
 *
 * There is one entry per MCH detection element, describing the 
 * basic features of all the pads of that detection element : 
 *
 * - electronic location : FEE board id (aka dual sampa id
 * and FEE channel id (aka dual sampa channel)
 * - geometric location : (x,y) positions (cm) within the detection element
 * - geometric size :  (dx,dy) half-sizes (cm)
 *
 */

int main(int argc, char** argv)
{
  std::string outputFileName;
  po::variables_map vm;
  po::options_description options("options");
  std::string defaultName = fmt::format("mch-mapping{}-tree.root", run2 ? "-run2" : "");
  // clang-format off
  options.add_options()
      ("help,h", "produce help message")
      ("outfile,o",po::value<std::string>(&outputFileName)->default_value(defaultName),"path to output file")
      ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(options);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << fmt::format("This program exports the MCH {}mapping to a Root tree\n",
                             run2 ? "(Run2 version) " : "");
    std::cout << " --outfile path to the output file containing the Root tree";
    std::cout << options << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  TFile fout(outputFileName.c_str(), "RECREATE");
  if (!fout.IsOpen()) {
    std::cout << "Cannot open output file " << outputFileName << "\n";
    exit(2);
  }

  TTree tree("mchpads", "Muon Chamber Pads");
  DePads depads;
  createPadBranches(tree, depads);

  forEachDetectionElement([&depads, &tree](int deid) {
    Segmentation seg{deid};
    if (seg.nofPads() > MaxNofPadsPerDE) {
      throw std::logic_error(fmt::format("Something is wrong : max number of pads should be below {} but got {}", MaxNofPadsPerDE, seg.nofPads()));
    }
    depads.nDePad = seg.nofPads();
    depads.deid = deid;
    for (auto padid = 0; padid < depads.nDePad; padid++) {
      depads.dsid[padid] = seg.padDualSampaId(padid);
      depads.dsch[padid] = seg.padDualSampaChannel(padid);
      depads.x[padid] = seg.padPositionX(padid);
      depads.y[padid] = seg.padPositionY(padid);
      depads.dx[padid] = seg.padSizeX(padid) / 2.0;
      depads.dy[padid] = seg.padSizeY(padid) / 2.0;
    }
    tree.Fill();
  });

  tree.Write();
  return 0;
}
