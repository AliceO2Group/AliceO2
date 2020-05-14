// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CruBufferCreator.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "DumpBuffer.h"
#include "Headers/RAWDataHeader.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawCommon/RDHManip.h"
#include "MCHRawEncoderPayload/DataBlock.h"
#include "MCHRawEncoderPayload/PayloadPaginator.h"
#include <boost/program_options.hpp>
#include "Framework/Logger.h"
#include <fstream>
#include <gsl/span>
#include <iostream>

namespace po = boost::program_options;
using namespace o2::mch::raw;
using V4 = o2::header::RAWDataHeaderV4;

namespace o2::header
{
extern std::ostream& operator<<(std::ostream&, const o2::header::RAWDataHeaderV4&);
}

void generateCxxFile(std::ostream& out, gsl::span<const std::byte> pages, bool userLogic)
{
  out << R"(// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "RefBuffers.h"
#include <array>
#include "MCHRawCommon/DataFormats.h"

)";

  const std::string arrayName = userLogic ? "REF_BUFFER_CRU_USERLOGIC_CHARGESUM" : "REF_BUFFER_CRU_BARE_CHARGESUM";

  out << fmt::format("extern std::array<const uint8_t,{}> {};\n",
                     pages.size_bytes(), arrayName);

  out << fmt::format("template <> gsl::span<const std::byte> REF_BUFFER_CRU<o2::mch::raw::{}, o2::mch::raw::ChargeSumMode>()\n", userLogic ? "UserLogicFormat" : "BareFormat");

  out << "{\n";

  out << fmt::format("return gsl::span<const std::byte>(reinterpret_cast<const std::byte*>(&{}[0]), {}.size());\n",
                     arrayName,
                     arrayName);

  out << "\n}\n";

  out
    << fmt::format("std::array<const uint8_t, {}> {}= {{", pages.size_bytes(), arrayName);

  out << R"(
// clang-format off
)";

  int i{0};
  for (auto v : pages) {
    out << fmt::format("0x{:02X}", v);
    if (i != pages.size_bytes() - 1) {
      out << ", ";
    }
    if (++i % 12 == 0) {
      out << "\n";
    }
  }

  out << R"(
// clang-format on
};
)";
}
void generate(gsl::span<std::byte> pages, bool userLogic)
{
  // set the chargesum mask for each rdh
  int n{0};
  o2::mch::raw::forEachRDH<V4>(pages,
                               [&](V4& rdh, gsl::span<std::byte>::size_type offset) {
                                 rdhFeeId(rdh, rdhFeeId(rdh) | 0x100);
                                 n++;
                               });

  generateCxxFile(std::cout, pages, userLogic);
  std::cout << "constexpr int generatedRDH=" << n << ";\n";
}

std::vector<std::byte> paginate(gsl::span<const std::byte> buffer,
                                bool userLogic)
{
  fair::Logger::SetConsoleSeverity("nolog");
  //fair::Logger::SetConsoleSeverity("debug");
  o2::raw::RawFileWriter fw;

  fw.setVerbosity(1);
  fw.setDontFillEmptyHBF(true);

  Solar2FeeLinkMapper solar2feelink;
  //using Solar2FeeLinkMapper = std::function<std::optional<FeeLinkId>(uint16_t solarId)>;

  if (userLogic) {
    solar2feelink = [](uint16_t solarId) -> std::optional<FeeLinkId> {
      static auto s2f = createSolar2FeeLinkMapper<ElectronicMapperGenerated>();
      auto f = s2f(solarId);
      if (!f.has_value()) {
        return std::nullopt;
      }
      return FeeLinkId(f->feeId(), 15);
    };
  } else {
    solar2feelink = createSolar2FeeLinkMapper<ElectronicMapperGenerated>();
  }

  auto tmpfile = "mch.cru.testbuffer.1.raw";
  {
    PayloadPaginator p(fw, tmpfile, solar2feelink);
    p(buffer);
    fw.close();
  }

  std::ifstream in(tmpfile, std::ifstream::binary);
  // get length of file:
  in.seekg(0, in.end);
  int length = in.tellg();
  in.seekg(0, in.beg);
  std::vector<std::byte> pages(length);

  // read data as a block:
  in.read(reinterpret_cast<char*>(&pages[0]), length);

  return pages;
}

int main(int argc, char** argv)
{
  po::options_description generic("options");
  bool userLogic{false};
  bool debugOnly{false};
  po::variables_map vm;

  // clang-format off
  generic.add_options()
      ("help,h", "produce help message")
      ("userLogic,u",po::bool_switch(&userLogic),"user logic format")
      ("debug,d",po::bool_switch(&debugOnly),"debug only");
  // clang-format on

  po::options_description cmdline;
  cmdline.add(generic);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << generic << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  po::notify(vm);

  uint32_t orbit = 12345;
  uint16_t bc = 678;
  std::vector<std::byte> buffer;
  if (userLogic) {
    buffer = test::CruBufferCreator<UserLogicFormat, ChargeSumMode>::makeBuffer(1, orbit, bc);
  } else {
    buffer = test::CruBufferCreator<BareFormat, ChargeSumMode>::makeBuffer(1, orbit, bc);
  }

  if (debugOnly) {
    auto solar2feelink = createSolar2FeeLinkMapper<ElectronicMapperGenerated>();
    forEachDataBlockRef(
      buffer, [&](const DataBlockRef& ref) {
        std::cout << ref << "\n";
        std::cout << solar2feelink(ref.block.header.solarId).value() << "\n";
        if (userLogic) {
          impl::dumpBuffer<UserLogicFormat>(ref.block.payload);
        } else {
          impl::dumpBuffer<BareFormat>(ref.block.payload);
        }
      });
  }

  std::vector<std::byte> pages;
  const o2::raw::HBFUtils& hbfutils = o2::raw::HBFUtils::Instance();
  o2::conf::ConfigurableParam::setValue<uint32_t>("HBFUtils", "orbitFirst", orbit);
  o2::conf::ConfigurableParam::setValue<uint16_t>("HBFUtils", "bcFirst", bc);
  pages = paginate(buffer, userLogic);

  if (debugOnly) {
    if (userLogic) {
      impl::dumpBuffer<UserLogicFormat>(pages);
    } else {
      impl::dumpBuffer<BareFormat>(pages);
    }
    return 0;
  }

  generate(pages, userLogic);
  return 0;
}
