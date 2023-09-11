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
/// \author mconcas@cern.ch
///
#include <unistd.h>

#include "../Shared/Kernels.h"
#define VERSION "version 0.3"

bool parseArgs(o2::benchmark::benchmarkOpts& conf, int argc, const char* argv[])
{
  namespace bpo = boost::program_options;
  bpo::variables_map vm;
  bpo::options_description options("Benchmark options");
  options.add_options()(
    "arbitrary,a", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{""}, ""), "Custom selected chunks syntax <p>:<s>. P is starting GB, S is the size in GB.")(
    "blockPool,b", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{"sb", "mb", "ab"}, "sb mb ab cb"), "Block pool strategy: single, multi, all or manual blocks.")(
    "chunkSize,c", bpo::value<float>()->default_value(1.f), "Size of scratch partitions (GB).")(
    "device,d", bpo::value<int>()->default_value(0), "Id of the device to run test on, EPN targeted.")(
    "threadPool,e", bpo::value<float>()->default_value(1.f), "Fraction of blockDim.x to use (aka: rounded fraction of thread pool).")(
    "freeMemFraction,f", bpo::value<float>()->default_value(0.95f), "Fraction of free memory to be allocated (min: 0.f, max: 1.f).")(
    "blocks,g", bpo::value<int>()->default_value(-1), "Number of blocks, manual mode. (g=-1: gridDim.x).")(
    "help,h", "Print help message.")(
    "inspect,i", "Inspect and dump chunk addresses.")(
    "threads,j", bpo::value<int>()->default_value(-1), "Number of threads per block, manual mode. (j=-1: blockDim.x).")(
    "kind,k", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{"char", "int", "ulong", "int4"}, "char int ulong int4"), "Test data type to be used.")(
    "launches,l", bpo::value<int>()->default_value(10), "Number of iterations in reading kernels.")(
    "mode,m", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{"seq", "con", "dis"}, "seq con dis"), "Mode: sequential, concurrent or distributed.")(
    "nruns,n", bpo::value<int>()->default_value(1), "Number of times each test is run.")(
    "outfile,o", bpo::value<std::string>()->default_value("benchmark_result"), "Output file name to store results.")(
    "prime,p", bpo::value<int>()->default_value(0), "Prime number to be used for the test.")(
    "raw,r", "Display raw output.")(
    "streams,s", bpo::value<int>()->default_value(8), "Size of the pool of streams available for concurrent tests.")(
    "test,t", bpo::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{"read", "write", "copy", "rread", "rwrite", "rcopy"}, "read write copy rread rwrite rcopy"), "Tests to be performed.")(
    "version,v", "Print version.")(
    "extra,x", "Print extra info for each available device.");
  try {
    bpo::store(parse_command_line(argc, argv, options), vm);
    if (vm.count("help")) {
      std::cout << options << std::endl;
      return false;
    }

    if (vm.count("version")) {
      std::cout << VERSION << std::endl;
      return false;
    }

    if (vm.count("extra")) {
      o2::benchmark::benchmarkOpts opts;
      o2::benchmark::GPUbenchmark<char> bm_dummy{opts};
      bm_dummy.printDevices();
      return false;
    }

    if (vm.count("inspect")) {
      conf.dumpChunks = true;
    }

    if (vm.count("raw")) {
      conf.raw = true;
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
  conf.threadPoolFraction = vm["threadPool"].as<float>();
  conf.numThreads = vm["threads"].as<int>();
  conf.numBlocks = vm["blocks"].as<int>();
  conf.chunkReservedGB = vm["chunkSize"].as<float>();
  conf.kernelLaunches = vm["launches"].as<int>();
  conf.nTests = vm["nruns"].as<int>();
  conf.streams = vm["streams"].as<int>();
  conf.prime = vm["prime"].as<int>();
  if ((conf.prime > 0 && !is_prime(conf.prime))) {
    std::cerr << "Invalid prime number: " << conf.prime << std::endl;
    exit(1);
  }

  conf.tests.clear();
  for (auto& test : vm["test"].as<std::vector<std::string>>()) {
    if (test == "read") {
      conf.tests.push_back(Test::Read);
    } else if (test == "write") {
      conf.tests.push_back(Test::Write);
    } else if (test == "copy") {
      conf.tests.push_back(Test::Copy);
    } else if (test == "rread") {
      if (!vm["prime"].as<int>()) {
        std::cerr << "Prime number must be specified for rread test." << std::endl;
        exit(1);
      }
      conf.tests.push_back(Test::RandomRead);
    } else if (test == "rwrite") {
      if (!vm["prime"].as<int>()) {
        std::cerr << "Prime number must be specified for rwrite test." << std::endl;
        exit(1);
      }
      conf.tests.push_back(Test::RandomWrite);
    } else if (test == "rcopy") {
      if (!vm["prime"].as<int>()) {
        std::cerr << "Prime number must be specified for rcopy test." << std::endl;
        exit(1);
      }
      conf.tests.push_back(Test::RandomCopy);
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
    } else if (mode == "dis") {
      conf.modes.push_back(Mode::Distributed);
    } else {
      std::cerr << "Unkonwn mode: " << mode << std::endl;
      exit(1);
    }
  }

  conf.pools.clear();
  for (auto& pool : vm["blockPool"].as<std::vector<std::string>>()) {
    if (pool == "sb") {
      conf.pools.push_back(KernelConfig::Single);
    } else if (pool == "mb") {
      conf.pools.push_back(KernelConfig::Multi);
    } else if (pool == "ab") {
      conf.pools.push_back(KernelConfig::All);
    } else if (pool == "cb") {
      if (vm["blocks"].as<int>() < 0) {
        std::cerr << "Manual pool setting requires --blocks or -g to be passed." << std::endl;
        exit(1);
      }
      conf.pools.push_back(KernelConfig::Manual);
    } else {
      std::cerr << "Unkonwn pool: " << pool << std::endl;
      exit(1);
    }
  }

  conf.testChunks.clear();
  for (auto& aChunk : vm["arbitrary"].as<std::vector<std::string>>()) {
    const size_t sep = aChunk.find(':');
    if (sep != std::string::npos) {
      conf.testChunks.emplace_back(std::stof(aChunk.substr(0, sep)), std::stof(aChunk.substr(sep + 1)));
    }
  }

  conf.dtypes = vm["kind"].as<std::vector<std::string>>();
  conf.outFileName = vm["outfile"].as<std::string>();

  return true;
}

int main(int argc, const char* argv[])
{
  std::cout << "Started benchmark with pid: " << getpid() << std::endl;
  o2::benchmark::benchmarkOpts opts;

  if (!parseArgs(opts, argc, argv)) {
    return -1;
  }

  for (auto& dtype : opts.dtypes) {
    if (dtype == "char") {
      o2::benchmark::GPUbenchmark<char> bm_char{opts};
      bm_char.run();
    } else if (dtype == "int") {
      o2::benchmark::GPUbenchmark<int> bm_int{opts};
      bm_int.run();
    } else if (dtype == "ulong") {
      o2::benchmark::GPUbenchmark<size_t> bm_size_t{opts};
      bm_size_t.run();
    } else if (dtype == "int4") {
      o2::benchmark::GPUbenchmark<int4> bm_size_t{opts};
      bm_size_t.run();
    } else {
      std::cerr << "Unkonwn data type: " << dtype << std::endl;
      exit(1);
    }
  }
  return 0;
}
