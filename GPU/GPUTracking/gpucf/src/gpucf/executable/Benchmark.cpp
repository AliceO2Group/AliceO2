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
    envFlags = std::make_unique<ClEnv::Flags>(required, optional); 
    cfflags = std::make_unique<CfCLIFlags>(required, optional); 

    digitFile = INIT_FLAG(
            StringFlag,
            required,
            "FILE",
            "File fo digits.",
            {'d', "digits"});

    outFile = INIT_FLAG(
            StringFlag,
            required,
            "FILE",
            "Benchmark results are written here.",
            {'o', "out"});

    iterations = INIT_FLAG(
            IntFlag,
            optional,
            "N",
            "How often each algorithm is run (default=10)",
            {'i', "iter"},
            10);
}

int Benchmark::mainImpl()
{
    SectorMap<std::vector<RawDigit>> rawdigits = read<RawDigit>(args::get(*digitFile));
    SectorMap<std::vector<Digit>> alldigits = Digit::bySector(rawdigits);
    digits = alldigits[0];

    log::Debug() << "Timebins = " << digits.back().time;

    ClusterFinderConfig config = cfflags->asConfig();

    ClEnv env(*envFlags, config);
    TimeCf exp("", args::get(*outFile), config, digits, args::get(*iterations));
    exp.run(env);

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
