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
///
/// \file benchmark.cxx
/// \author mconcas@cern.ch
/// \brief configuration widely inspired/copied by SimConfig
#include "Shared/Kernels.h"

bool parseArgs(o2::benchmark::benchmarkOpts& conf, int argc, const char* argv[])
{
  namespace bpo = boost::program_options;
  bpo::variables_map vm;
  bpo::options_description options("Benchmark options");
  options.add_options()(
    "help,h", "Print help message.")(
    "chunkSize,c", bpo::value<float>()->default_value(1.f), "Size of scratch partitions (GB).")(
    "freeMemFraction,f", bpo::value<float>()->default_value(0.95f), "Fraction of free memory to be allocated (min: 0.f, max: 1.f).")(
    "iterations,i", bpo::value<size_t>()->default_value(50), "Number of iterations in reading kernels.");
  try {
    bpo::store(parse_command_line(argc, argv, options), vm);
    if (vm.count("help")) {
      std::cout << options << std::endl;
      return false;
    }

    bpo::notify(vm);
  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing command line arguments. Available options:\n";

    std::cerr << options << std::endl;
    return false;
  }

  conf.freeMemoryFractionToAllocate = vm["freeMemFraction"].as<float>();
  conf.partitionSizeGB = vm["chunkSize"].as<float>();
  conf.iterations = vm["iterations"].as<size_t>();

  return true;
}

int main(int argc, const char* argv[])
{

  o2::benchmark::benchmarkOpts opts;
  if (argc > 1) {
    if (!parseArgs(opts, argc, argv)) {
      return -1;
    }
  }

  o2::benchmark::GPUbenchmark<char> bm_char{opts};
  bm_char.run();
  o2::benchmark::GPUbenchmark<size_t> bm_size_t{opts};
  bm_size_t.run();
  o2::benchmark::GPUbenchmark<int> bm_int{opts};
  bm_int.run();

  return 0;
}
