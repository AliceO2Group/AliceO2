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
    "device,d", bpo::value<int>()->default_value(0), "Id of the device to run test on, EPN targeted.")(
    "test,t", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{"read", "write", "copy"}, "read, write, copy"), "Tests to be performed.")(
    "mode,m", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{"seq", "con"}, "seq, con"), "Mode: sequential or concurrent.")(
    "pool,p", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{"sb, mb"}, "sb, mb"), "Pool strategy: single or multi blocks.")(
    "chunkSize,c", bpo::value<float>()->default_value(1.f), "Size of scratch partitions (GB).")(
    "freeMemFraction,f", bpo::value<float>()->default_value(0.95f), "Fraction of free memory to be allocated (min: 0.f, max: 1.f).")(
    "launches,l", bpo::value<int>()->default_value(10), "Number of iterations in reading kernels.")(
    "nruns,n", bpo::value<int>()->default_value(1), "Number of times each test is run.");
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

  conf.deviceId = vm["device"].as<int>();
  conf.freeMemoryFractionToAllocate = vm["freeMemFraction"].as<float>();
  conf.chunkReservedGB = vm["chunkSize"].as<float>();
  conf.nRegions = vm["regions"].as<int>();
  conf.kernelLaunches = vm["launches"].as<int>();
  conf.nTests = vm["nruns"].as<int>();

  conf.tests.clear();
  for (auto& test : vm["test"].as<std::vector<std::string>>()) {
    if (test == "read") {
      conf.tests.push_back(Test::Read);
    } else if (test == "write") {
      conf.tests.push_back(Test::Write);
    } else if (test == "copy") {
      conf.tests.push_back(Test::Copy);
    } else {
      std::cerr << "Unkonwn test: " << test << std::endl;
      exit(1);
    }
  }

  conf.modes.clear();
  for (auto& mode : vm["mode"].as<std::vector<std::string>>()) {
    if (mode == "seq") {
      conf.modes.push_back(Mode::Sequential);
    } else if (mode == "con") {
      conf.modes.push_back(Mode::Concurrent);
    } else {
      std::cerr << "Unkonwn mode: " << mode << std::endl;
      exit(1);
    }
  }

  conf.pools.clear();
  for (auto& pool : vm["pool"].as<std::vector<std::string>>()) {
    if (pool == "sb") {
      conf.pools.push_back(SplitLevel::Blocks);
    } else if (pool == "mb") {
      conf.pools.push_back(SplitLevel::Threads);
    } else {
      std::cerr << "Unkonwn pool: " << pool << std::endl;
      exit(1);
    }
  }

  return true;
}

using o2::benchmark::ResultWriter;

int main(int argc, const char* argv[])
{

  o2::benchmark::benchmarkOpts opts;

  if (!parseArgs(opts, argc, argv)) {
    return -1;
  }

  std::shared_ptr<ResultWriter> writer = std::make_shared<ResultWriter>(std::to_string(opts.deviceId) + "_benchmark_results.root");

  o2::benchmark::GPUbenchmark<char> bm_char{opts, writer};
  bm_char.run();
  o2::benchmark::GPUbenchmark<int> bm_int{opts, writer};
  bm_int.run();
  o2::benchmark::GPUbenchmark<size_t> bm_size_t{opts, writer};
  bm_size_t.run();

  // save results
  writer.get()->saveToFile();

  return 0;
}
