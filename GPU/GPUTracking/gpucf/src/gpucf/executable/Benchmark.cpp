#include "Benchmark.h"

#include <gpucf/common/log.h>
#include <gpucf/common/RawLabel.h>
#include <gpucf/common/serialization.h>
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

    SectorMap<std::vector<RawDigit>> rawdigits = read<RawDigit>(args::get(*digitFile));
    SectorMap<std::vector<Digit>> alldigits = Digit::bySector(rawdigits);
    digits = alldigits[0];

    log::Debug() << "Timebins = " << digits.back().time;

    registerExperiments();

    runExperiments();

    return 0;
}

void Benchmark::registerExperiments()
{
    {
        ClusterFinderConfig scratchpad;
        scratchpad.halfs = true;
        scratchpad.layout = ChargemapLayout::Tiling4x4;
        scratchpad.clusterbuilder = ClusterBuilder::ScratchPad;
        experiments.emplace_back(
                new TimeCf(
                        "Load charges into scratchpad",
                        "scratchpad4x4.json",
                        scratchpad,
                        digits,
                        iterations->Get(),
                        baseDir));
    }

    {
        ClusterFinderConfig scratchpad;
        scratchpad.halfs = true;
        scratchpad.layout = ChargemapLayout::Tiling4x8;
        scratchpad.clusterbuilder = ClusterBuilder::ScratchPad;
        experiments.emplace_back(
                new TimeCf(
                        "Load charges into scratchpad",
                        "scratchpad4x8.json",
                        scratchpad,
                        digits,
                        iterations->Get(),
                        baseDir));
    }

    {
        ClusterFinderConfig scratchpad;
        scratchpad.halfs = true;
        scratchpad.layout = ChargemapLayout::Tiling8x4;
        scratchpad.clusterbuilder = ClusterBuilder::ScratchPad;
        experiments.emplace_back(
                new TimeCf(
                        "Load charges into scratchpad",
                        "scratchpad8x4.json",
                        scratchpad,
                        digits,
                        iterations->Get(),
                        baseDir));
    }

    {
        ClusterFinderConfig timeMajorLayout;
        timeMajorLayout.layout = ChargemapLayout::TimeMajor;
        experiments.emplace_back(
                new TimeCf(
                        "Chargemap with tiling layout", 
                        "timemajor.json",
                        timeMajorLayout,
                        digits,
                        iterations->Get(), 
                        baseDir));
    }

    /* { */
    /*     ClusterFinderConfig tilingLayout; */
    /*     tilingLayout.layout = ChargemapLayout::Tiling4x4; */
    /*     experiments.emplace_back( */
    /*             new TimeCf( */
    /*                     "Chargemap with tiling layout", */ 
    /*                     "tilingLayout.json", */
    /*                     tilingLayout, */
    /*                     digits, */
    /*                     iterations->Get(), */ 
    /*                     baseDir)); */
    /* } */

    {
        ClusterFinderConfig padMajorLayout;
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

    /* { */
    /*     ClusterFinderConfig halfs; */
    /*     halfs.halfs = true; */
    /*     experiments.emplace_back( */
    /*             new TimeCf( */
    /*                     "Chargemap storing half charges.", */
    /*                     "halfs.json", */
    /*                     halfs, */
    /*                     digits, */
    /*                     iterations->Get(), */
    /*                     baseDir)); */
    /* } */

    /* { */
    /*     ClusterFinderConfig halfsPadMajor; */
    /*     halfsPadMajor.halfs = true; */
    /*     halfsPadMajor.layout = ChargemapLayout::PadMajor; */
    /*     experiments.emplace_back( */
    /*             new TimeCf( */
    /*                     "Chargemap storing half charges (pad major layout)", */
    /*                     "halfsPadMajor.json", */
    /*                     halfsPadMajor, */
    /*                     digits, */
    /*                     iterations->Get(), */
    /*                     baseDir)); */
        
    /* } */

    /* { */
    /*     ClusterFinderConfig halfs4x8Tiling; */
    /*     halfs4x8Tiling.halfs = true; */
    /*     halfs4x8Tiling.layout = ChargemapLayout::Tiling4x8; */
    /*     experiments.emplace_back( */
    /*             new TimeCf( */
    /*                     "Chargemap storing half charges (4x8 tiling layout)", */
    /*                     "halfs4x8Tiling.json", */
    /*                     halfs4x8Tiling, */
    /*                     digits, */
    /*                     iterations->Get(), */
    /*                     baseDir)); */
        
    /* } */

    /* { */
    /*     ClusterFinderConfig halfs8x4Tiling; */
    /*     halfs8x4Tiling.halfs = true; */
    /*     halfs8x4Tiling.layout = ChargemapLayout::Tiling8x4; */
    /*     experiments.emplace_back( */
    /*             new TimeCf( */
    /*                     "Chargemap storing half charges (8x4 tiling layout)", */
    /*                     "halfs8x4Tiling.json", */
    /*                     halfs8x4Tiling, */
    /*                     digits, */
    /*                     iterations->Get(), */
    /*                     baseDir)); */
    /* } */


}

void Benchmark::runExperiments()
{
    for (auto &experiment : experiments)
    {
        ClEnv env(*envFlags, experiment->getConfig());
        experiment->run(env); 
    }
}

// vim: set ts=4 sw=4 sts=4 expandtab:
