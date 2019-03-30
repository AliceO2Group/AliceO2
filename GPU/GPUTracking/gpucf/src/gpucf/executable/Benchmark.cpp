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
        tilingLayout.layout = ChargemapLayout::Tiling4x4;
        experiments.emplace_back(
                new TimeCf(
                        "Chargemap with tiling layout", 
                        "tilingLayout.json",
                        tilingLayout,
                        digits,
                        iterations->Get(), 
                        baseDir));
    }

    {
        GPUClusterFinder::Config padMajorLayout;
        padMajorLayout.layout = ChargemapLayout::PadMajor;
        experiments.emplace_back(
                new TimeCf(
                        "Chargemap with pad major layout",
                        "padMajor.json",
                        padMajorLayout,
                        digits,
                        iterations->Get(),
                        baseDir));
    }

    {
        GPUClusterFinder::Config halfs;
        halfs.halfPrecisionCharges = true;
        experiments.emplace_back(
                new TimeCf(
                        "Chargemap storing half charges.",
                        "halfs.json",
                        halfs,
                        digits,
                        iterations->Get(),
                        baseDir));
    }

    {
        GPUClusterFinder::Config halfsPadMajor;
        halfsPadMajor.halfPrecisionCharges = true;
        halfsPadMajor.layout = ChargemapLayout::PadMajor;
        experiments.emplace_back(
                new TimeCf(
                        "Chargemap storing half charges (pad major layout)",
                        "halfsPadMajor.json",
                        halfsPadMajor,
                        digits,
                        iterations->Get(),
                        baseDir));
        
    }

    {
        GPUClusterFinder::Config halfs4x8Tiling;
        halfs4x8Tiling.halfPrecisionCharges = true;
        halfs4x8Tiling.layout = ChargemapLayout::Tiling4x8;
        experiments.emplace_back(
                new TimeCf(
                        "Chargemap storing half charges (4x8 tiling layout)",
                        "halfs4x8Tiling.json",
                        halfs4x8Tiling,
                        digits,
                        iterations->Get(),
                        baseDir));
        
    }

    {
        GPUClusterFinder::Config halfs8x4Tiling;
        halfs8x4Tiling.halfPrecisionCharges = true;
        halfs8x4Tiling.layout = ChargemapLayout::Tiling8x4;
        experiments.emplace_back(
                new TimeCf(
                        "Chargemap storing half charges (8x4 tiling layout)",
                        "halfs8x4Tiling.json",
                        halfs8x4Tiling,
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
