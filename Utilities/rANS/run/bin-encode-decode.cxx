// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   bin-encode-decode.cpp
/// @author Michael Lettrich
/// @since  2020-06-22
/// @brief  benchmark encode/decode using rans on binary data.

#include "rANS/rans.h"
#include "rANS/utils.h"

#include <boost/program_options.hpp>

#include <fairlogger/Logger.h>

namespace bpo = boost::program_options;

#ifndef SOURCE_T
#define SOURCE_T uint8_t
#endif

using source_t = SOURCE_T;
using coder_t = uint64_t;
using stream_t = uint32_t;
static const uint REPETITIONS = 5;

template <typename T>
void readFile(const std::string& filename, std::vector<T>* tokens)
{
  std::ifstream is(filename, std::ios_base::binary | std::ios_base::in);
  if (is) {
    // get length of file:
    is.seekg(0, is.end);
    size_t length = is.tellg();
    is.seekg(0, is.beg);

    // reserve size of tokens
    if (!tokens) {
      throw std::runtime_error("Cannot read file into nonexistent vector");
    }

    if (length % sizeof(T)) {
      throw std::runtime_error("Filesize is not a multiple of datatype.");
    }
    // size the vector appropriately
    size_t num_elems = length / sizeof(T);
    tokens->resize(num_elems);

    // read data as a block:
    is.read(reinterpret_cast<char*>(tokens->data()), length);
    is.close();
  }
}

int main(int argc, char* argv[])
{

  bpo::options_description options("Allowed options");
  // clang-format off
  options.add_options()
    ("help,h", "print usage message")
    ("file,f",bpo::value<std::string>(), "file to compress")
    ("samples,s",bpo::value<uint32_t>(), "how often to run benchmark")
    ("bits,b",bpo::value<uint32_t>(), "resample dictionary to N Bits")
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

  const uint32_t probabilityBits = [&]() {
    if (vm.count("bits")) {
      return vm["bits"].as<uint32_t>();
    } else {
      return 0u;
    }
  }();

  const uint32_t repetitions = [&]() {
    if (vm.count("samples")) {
      return vm["samples"].as<uint32_t>();
    } else {
      return REPETITIONS;
    }
  }();

  if (vm.count("log_severity")) {
    fair::Logger::SetConsoleSeverity(vm["log_severity"].as<std::string>().c_str());
  }
  for (size_t i = 0; i < repetitions; i++) {
    LOG(info) << "repetion: " << i;
    std::vector<source_t> tokens;
    readFile(filename, &tokens);

    o2::rans::FrequencyTable frequencies;
    frequencies.addSamples(std::begin(tokens), std::end(tokens));

    std::vector<stream_t> encoderBuffer;
    const o2::rans::Encoder64<source_t> encoder{frequencies, probabilityBits};
    encoder.process(std::begin(tokens), std::end(tokens), std::back_inserter(encoderBuffer));

    std::vector<source_t> decoderBuffer(tokens.size());
    [&]() {
      o2::rans::Decoder64<source_t> decoder{frequencies, probabilityBits};
      decoder.process(encoderBuffer.end(), decoderBuffer.begin(), std::distance(std::begin(tokens), std::end(tokens)));
    }();

    if (std::memcmp(tokens.data(), decoderBuffer.data(),
                    tokens.size() * sizeof(source_t))) {
      LOG(error) << "Decoder failed tests";
    } else {
      LOG(info) << "Decoder passed tests";
    }
  }
}
