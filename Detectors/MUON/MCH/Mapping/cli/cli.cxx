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

#include "MCHMappingInterface/Segmentation.h"
#include <iostream>
#include <fmt/format.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

void dumpPad(const o2::mch::mapping::Segmentation& seg, int padId)
{
  std::cout << fmt::format("DE {:4d} DS {:4d} CH {:2d} PADID {:6d} X {:7.2f} Y {:7.2f} SX{:7.2f} SY {:7.2f}\n",
                           seg.detElemId(),
                           seg.padDualSampaId(padId),
                           seg.padDualSampaChannel(padId),
                           padId,
                           seg.padPositionX(padId),
                           seg.padPositionY(padId),
                           seg.padSizeX(padId),
                           seg.padSizeY(padId));
}

void dumpPad(const o2::mch::mapping::Segmentation& seg, int dsId, int dsCh)
{
  auto padId = seg.findPadByFEE(dsId, dsCh);
  if (seg.isValid(padId)) {
    dumpPad(seg, padId);
  } else {
    std::cout << fmt::format("DE {:4d} DS {:4d} CH {:2d} PADID {:6d} (channel not connected to an actual pad)\n",
                             seg.detElemId(), dsId, dsCh, padId);
  }
}

void dumpDualSampa(const o2::mch::mapping::Segmentation& seg, int dualSampaId)
{
  for (auto ch = 0; ch < 64; ch++) {
    dumpPad(seg, dualSampaId, ch);
  }
}

void dumpDetectionElement(const o2::mch::mapping::Segmentation& seg)
{
  seg.forEachDualSampa([&](int dualSampaId) {
    dumpDualSampa(seg, dualSampaId);
  });
}

int main(int argc, char** argv)
{
  int deId;
  int dsId;
  int dsCh;
  int padId;
  po::variables_map vm;
  po::options_description generic("Generic options");

  // clang-format off
  generic.add_options()
      ("help,h", "produce help message")
      ("de,d",po::value<int>(&deId)->required(),"detection element id")
      ("ds,s",po::value<int>(&dsId),"dual sampa id")
      ("ch,c",po::value<int>(&dsCh),"dual sampa ch")
      ("padid,p",po::value<int>(&padId),"padid")
      ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(generic);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << "This program printout MCH basic mapping information.\n";
    std::cout << " --de # : for a full detection element\n";
    std::cout << " --de # --ds # : for one dual sampa of a detection element\n";
    std::cout << " --de # --ds # --ch # : for one channel of a one dual sampa of a detection element\n";
    std::cout << " --de # --padid # : for one pad of a one dual sampa of a detection element\n";
    std::cout << "Pad sizes and positions are reported in centimeters\n";
    std::cout << generic << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  o2::mch::mapping::Segmentation seg(deId);

  if (vm.count("padid")) {
    dumpPad(seg, padId);
    return 0;
  }

  if (vm.count("ds") && vm.count("ch")) {
    dumpPad(seg, dsId, dsCh);
    return 0;
  }

  if (vm.count("ds")) {
    dumpDualSampa(seg, dsId);
    return 0;
  }

  dumpDetectionElement(seg);

  return 0;
}
