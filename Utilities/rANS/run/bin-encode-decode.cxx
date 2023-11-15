// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   bin-encode-decode.cpp
/// @author Michael Lettrich
/// @brief  run rANS encoder and decoder based on 8,16 and 32 bit binary input.

#include <rANS/factory.h>
#include <rANS/histogram.h>
#include <rANS/encode.h>
#include <rANS/decode.h>

#include <boost/program_options.hpp>
#include <algorithm>
#include <iostream>

#include <fairlogger/Logger.h>

namespace bpo = boost::program_options;

#ifndef SOURCE_T
#define SOURCE_T uint8_t
#endif

using source_type = SOURCE_T;
using stream_type = uint32_t;
inline constexpr size_t NSTREAMS = 2;
inline constexpr size_t LOWER_BOUND = 31;

template <typename T>
std::vector<T> readFile(const std::string& filename)
{
  std::vector<T> tokens{};
  std::ifstream is(filename, std::ios_base::binary | std::ios_base::in);
  if (is) {
    // get length of file:
    is.seekg(0, is.end);
    size_t length = is.tellg();
    is.seekg(0, is.beg);

    if (length % sizeof(T)) {
      throw o2::rans::IOError("Filesize is not a multiple of datatype.");
    }
    // size the vector appropriately
    tokens.resize(length / sizeof(T));

    // read data as a block:
    is.read(reinterpret_cast<char*>(tokens.data()), length);
    is.close();
  }
  return tokens;
}

int main(int argc, char* argv[])
{

  using namespace o2::rans;

  bpo::options_description options("Allowed options");
  // clang-format off
  options.add_options()
    ("help,h", "print usage message")
    ("file,f",bpo::value<std::string>(), "file to compress")
    ("log_severity,l",bpo::value<std::string>(), "severity of FairLogger");
  // clang-format on

  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(argc, argv, options), vm);
  bpo::notify(vm);

  if (vm.count("help")) {
    std::cout << options << "\n";
    return 0;
  }

  const std::string filename = [&]() {
    if (vm.count("file")) {
      return vm["file"].as<std::string>();
    } else {
      LOG(error) << "missing path to input file";
      exit(1);
    }
  }();

  if (vm.count("log_severity")) {
    fair::Logger::SetConsoleSeverity(vm["log_severity"].as<std::string>().c_str());
  }

  std::vector<source_type> tokens = readFile<source_type>(filename);

  // build encoders
  auto histogram = makeDenseHistogram::fromSamples(tokens.begin(), tokens.end());
  Metrics<source_type> metrics{histogram};
  auto renormedHistogram = renorm(std::move(histogram), metrics);
  auto encoder = makeDenseEncoder<CoderTag::SingleStream, NSTREAMS, LOWER_BOUND>::fromRenormed(renormedHistogram);
  auto decoder = makeDecoder<LOWER_BOUND>::fromRenormed(renormedHistogram);

  std::vector<stream_type> encoderBuffer;
  std::vector<source_type> decodeBuffer(tokens.size(), 0);
  std::vector<source_type> incompressibleSymbols;

  if (renormedHistogram.hasIncompressibleSymbol()) {
    LOG(info) << "With incompressible symbols";
    [[maybe_unused]] auto res = encoder.process(tokens.begin(), tokens.end(), std::back_inserter(encoderBuffer), std::back_inserter(incompressibleSymbols));
    LOGP(info, "nIncompressible {}", incompressibleSymbols.size());
    decoder.process(encoderBuffer.end(), decodeBuffer.begin(), tokens.size(), NSTREAMS, incompressibleSymbols.end());
  } else {
    LOG(info) << "Without incompressible symbols";
    encoder.process(std::begin(tokens), std::end(tokens), std::back_inserter(encoderBuffer));
    decoder.process(encoderBuffer.end(), decodeBuffer.begin(), tokens.size(), NSTREAMS);
  }

  size_t pos = 0;
  if (std::equal(tokens.begin(), tokens.end(), decodeBuffer.begin(), decodeBuffer.end(),
                 [&pos](const auto& a, const auto& b) {
                   const bool cmp = a == b;
                   if (!cmp) {
                     LOG(error) << fmt::format("[{}] {} != {}", pos, a, b);
                   }
                   ++pos;
                   return cmp;
                 })) {
    LOG(info) << "Decoder passed tests";
  } else {
    LOG(error) << "Decoder failed tests";
  }
};
