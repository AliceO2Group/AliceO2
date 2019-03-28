#include "Benchmark.h"

#include <gpucf/common/DataSet.h>
#include <gpucf/common/log.h>
#include <gpucf/errors/FileErrors.h>

#include <gpucf/experiments/TimeCf.h>


using namespace gpucf;
namespace fs = filesystem;


Benchmark::Benchmark()
    : Executable("Run all benchmarks.")
{
}

void Benchmark::setupFlags(args::Group &required, args::Group &optional)
{
    envFlags  = std::make_unique<ClEnv::Flags>(required, optional); 

    digitFile = INIT_FLAG(
            StringFlag,
            required,
            "FILE",
            "File fo digits.",
            {'d', "digits"});

    benchmarkDir = INIT_FLAG(
            StringFlag,
            required,
            "DIR",
            "Timing measurements are written here.",
            {'o', "out"});

    iterations = INIT_FLAG(
            IntFlag,
            optional,
            "N",
            "How often each algorithm is run (default=10)",
            {'o', "out"},
            10);
}

int Benchmark::mainImpl()
{
    baseDir = benchmarkDir->Get();
    if (!baseDir.is_directory())
    {
        throw DirectoryNotFoundError(baseDir);
    }

    DataSet digitSet;
    digitSet.read(args::get(*digitFile));

    digits = digitSet.deserialize<Digit>();

    registerExperiments();

    ClEnv env(*envFlags); 
    runExperiments(env);

    return 0;
}

void Benchmark::registerExperiments()
{
    {
        GPUClusterFinder::Config naiveConfig;
        naiveConfig.usePackedDigits = false;
        experiments.emplace_back(
                new TimeCf(
                        "naive cluster finder", 
                        "paddedClusterFinder.json",
                        naiveConfig,
                        digits, 
                        iterations->Get(), 
                        baseDir));
    }

    {
        GPUClusterFinder::Config packedDigitsConf;
        packedDigitsConf.usePackedDigits = true;
        experiments.emplace_back(
                new TimeCf(
                        "packed digits cluster finder", 
                        "packedClusterFinder.json",
                        packedDigitsConf,
                        digits, 
                        iterations->Get(), 
                        baseDir));
    }

    {
        GPUClusterFinder::Config multipleChunks;
        multipleChunks.usePackedDigits = true;
        multipleChunks.chunks = 4;
        experiments.emplace_back(
                new TimeCf(
                        "Parallel cluster finder", 
                        "parallelClusterFinder.json",
                        multipleChunks,
                        digits, 
                        iterations->Get(), 
                        baseDir));
    }

    {
        GPUClusterFinder::Config tilingLayout;
        tilingLayout.usePackedDigits = true;
        tilingLayout.useTilingLayout = true;
        experiments.emplace_back(
                new TimeCf(
                        "Chargemap with tiling layout", 
                        "tilingLayout.json",
                        tilingLayout,
                        digits,
                        iterations->Get(), 
                        baseDir));
    }
}


void Benchmark::runExperiments(ClEnv &env)
{
    for (auto &experiment : experiments)
    {
        ClEnv currEnv = env;
        experiment->run(currEnv); 
    }
}

// vim: set ts=4 sw=4 sts=4 expandtab:
