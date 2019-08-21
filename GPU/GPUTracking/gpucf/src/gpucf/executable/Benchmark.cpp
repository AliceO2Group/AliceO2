#include "Benchmark.h"

#include <gpucf/common/log.h>
#include <gpucf/common/RawLabel.h>
#include <gpucf/common/serialization.h>
#include <gpucf/errors/FileErrors.h>
#include <gpucf/experiments/TimeCf.h>

#include <cstdlib>


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

    sorting = INIT_FLAG(
            StringFlag,
            optional,
            "ORDER",
            "Order of sorted digits (time, pad, full, random) (default=time)",
            {'c', "comp"},
            "time");
}

int Benchmark::mainImpl()
{
    SectorMap<std::vector<RawDigit>> rawdigits = read<RawDigit>(args::get(*digitFile));
    SectorMap<std::vector<Digit>> alldigits = Digit::bySector(rawdigits);
    digits = alldigits[0];

    log::Debug() << "Timebins = " << digits.back().time;

    std::string sort = args::get(*sorting);

    if (sort == "time")
    {
        // digits are already time sorted
    }
    else if (sort == "pad")
    {
        shuffle(digits);         
        std::sort(digits.begin(), digits.end(),
                [] (const Digit &d1, const Digit &d2) {
                    if (d1.row < d2.row) return true;
                    if (d1.row > d2.row) return false;
                    if (d1.pad < d2.pad) return true;
                    return false;
                }
        );
    }
    else if (sort == "full")
    {
        shuffle(digits);         
        std::sort(digits.begin(), digits.end(),
                [] (const Digit &d1, const Digit &d2) {
                    if (d1.time < d2.time) return true;
                    if (d1.time > d2.time) return false;
                    if (d1.row < d2.row) return true;
                    if (d1.row > d2.row) return false;
                    if (d1.pad < d2.pad) return true;
                    return false;
                }
        );

    }
    else if (sort == "random")
    {
        shuffle(digits);         
    }
    else
    {
        log::Error() << "Unknown sorting order " << sort;
        showHelpAndExit();
    }

    ClusterFinderConfig config = cfflags->asConfig();

    ClEnv env(*envFlags, config);
    TimeCf exp("", args::get(*outFile), config, digits, args::get(*iterations));
    exp.run(env);

    return 0;
}

void Benchmark::shuffle(nonstd::span<Digit> digits)
{
    ASSERT(digits.size() < RAND_MAX);
    for (size_t i = digits.size()-1; i > 0; i--)
    {
        size_t j = rand() % i;
        std::swap(digits[i], digits[j]);
    }
}

// vim: set ts=4 sw=4 sts=4 expandtab:
