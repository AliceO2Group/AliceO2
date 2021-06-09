// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <iostream>
#include <fmt/format.h>
#include <boost/program_options.hpp>
#include "MCHRawElecMap/Mapper.h"
#include <stdexcept>

namespace po = boost::program_options;

template <typename ELECMAP>
int dump(const o2::mch::raw::DsElecId& dsElecId,
         const o2::mch::raw::DsDetId& dsDetId)
{
  auto solar2fee = o2::mch::raw::createSolar2FeeLinkMapper<ELECMAP>();
  auto feelink = solar2fee(dsElecId.solarId());
  if (!feelink.has_value()) {
    std::cout << "Could not get FeeLinkId for solarId " << dsElecId.solarId() << "\n";
    return 4;
  }
  std::cout << fmt::format("{} (elinkId {:d}) [ {} ] {}\n",
                           o2::mch::raw::asString(dsElecId),
                           dsElecId.elinkId(),
                           o2::mch::raw::asString(feelink.value()),
                           o2::mch::raw::asString(dsDetId));
  return 0;
}

template <typename ELECMAP>
int convertElec2Det(uint16_t solarId, uint8_t groupId, uint8_t indexId)
{
  try {
    std::cout << fmt::format("solarId {} groupId {} indexId {}\n",
                             solarId, groupId, indexId);
    o2::mch::raw::DsElecId dsElecId(solarId, groupId, indexId);
    auto elec2det = o2::mch::raw::createElec2DetMapper<ELECMAP>();
    auto dsDetId = elec2det(dsElecId);
    if (!dsDetId.has_value()) {
      std::cout << o2::mch::raw::asString(dsElecId) << " is not (yet?) known to the electronic mapper\n";
      return 3;
    }
    return dump<ELECMAP>(dsElecId, dsDetId.value());
  } catch (const std::exception& e) {
    std::cout << e.what() << "\n";
    return 4;
  }
}

template <typename ELECMAP>
int convertDet2Elec(int deId, int dsId)
{
  try {
    o2::mch::raw::DsDetId dsDetId(deId, dsId);
    auto det2elec = o2::mch::raw::createDet2ElecMapper<ELECMAP>();
    auto dsElecId = det2elec(dsDetId);
    if (!dsElecId.has_value()) {
      std::cout << o2::mch::raw::asString(dsDetId) << " is not (yet?) known to the electronic mapper\n";
      return 3;
    }
    return dump<ELECMAP>(dsElecId.value(), dsDetId);
  } catch (const std::exception& e) {
    std::cout << e.what() << "\n";
    return 5;
  }
}

int usage(const po::options_description& generic)
{
  std::cout << "This program converts MCH electronic mapping information to detector mapping information\n";
  std::cout << "(solarId,groupId,indexId)->(dsId,deId)";
  std::cout << "As well as the reverse operation\n";
  std::cout << generic << "\n";
  return 2;
}

int main(int argc, char** argv)
{
  int solarId;
  int groupId;
  int indexId;
  int deId;
  int dsId;
  bool dummyElecMap;

  po::variables_map vm;
  po::options_description generic("Generic options");

  // clang-format off
  generic.add_options()
      ("help,h", "produce help message")
      ("solarId,s",po::value<int>(&solarId),"solar id")
      ("groupId,g",po::value<int>(&groupId),"group id")
      ("indexId,i",po::value<int>(&indexId),"index id")
      ("dsId,d",po::value<int>(&dsId),"dual sampa id")
      ("deId,e",po::value<int>(&deId),"detection element id")
      ("dummy-elecmap",po::value<bool>(&dummyElecMap)->default_value(false),"use dummy electronic mapping (only for debug!)")
      ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(generic);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    return usage(generic);
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  if (vm.count("solarId") && vm.count("groupId") && vm.count("indexId")) {
    if (dummyElecMap) {
      return convertElec2Det<o2::mch::raw::ElectronicMapperDummy>(static_cast<uint16_t>(solarId),
                                                                  static_cast<uint8_t>(groupId),
                                                                  static_cast<uint8_t>(indexId));
    } else {
      return convertElec2Det<o2::mch::raw::ElectronicMapperGenerated>(static_cast<uint16_t>(solarId),
                                                                      static_cast<uint8_t>(groupId),
                                                                      static_cast<uint8_t>(indexId));
    }
  } else if (vm.count("deId") && vm.count("dsId")) {
    if (dummyElecMap) {
      return convertDet2Elec<o2::mch::raw::ElectronicMapperDummy>(deId, dsId);
    } else {
      return convertDet2Elec<o2::mch::raw::ElectronicMapperGenerated>(deId, dsId);
    }
  } else {
    std::cout << "Incorrect mix of options\n";
    return usage(generic);
  }

  return 0;
}
