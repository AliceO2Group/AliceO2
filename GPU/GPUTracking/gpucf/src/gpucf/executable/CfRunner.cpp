#include "CfRunner.h"

#include <gpucf/algorithms/GPUClusterFinder.h>
#include <gpucf/algorithms/ReferenceClusterFinder.h>
#include <gpucf/common/DataSet.h>
#include <gpucf/common/log.h>


using namespace gpucf;


CfRunner::CfRunner()
    : Executable("Runs the GPU cluster finder.")
{
}

void CfRunner::setupFlags(args::Group &required, args::Group &optional)
{
    envFlags  = std::make_unique<ClEnv::Flags>(required, optional); 

    digitFile = INIT_FLAG(
            StringFlag,
            required,
            "FILE",
            "File of digits",
            {'d', "digits"});

    clusterResultFile = INIT_FLAG(
            StringFlag,
            required,
            "FILE",
            "Computed clusters are written here.",
            {'o', "out"});

    peakFile = INIT_FLAG(
            StringFlag,
            optional,
            "FILE",
            "Cluster peaks are writtern here.",
            {'p', "peaks"});


    cfconfig = std::make_unique<args::Group>(
            optional,
            "Cluster finder config");

    tiling4x4 = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Use 4x4 tiling layout",
            {"tiling4x4"});

    tiling4x8 = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Use 4x8 tiling layout",
            {"tiling4x8"});

    tiling8x4 = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Use 8x4 tiling layout",
            {"tiling8x4"});

    scratchpad = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Load charges into scratchpad before building cluster.",
            {"scratchpad"});

    padMajor = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Use pad major in charge map",
            {"padMajor"});

    halfs = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Store charges in charge map as halfs.",
            {"halfs"});

    splitCharges = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Split charges among neighboring clusters.",
            {"split"});

    cpu = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Run cluster finder on cpu.",
            {"cpu"});
}

int CfRunner::mainImpl()
{
    ClEnv env(*envFlags); 

    DataSet digitSet;
    digitSet.read(args::get(*digitFile));

    std::vector<Digit> digits = digitSet.deserialize<Digit>();

    ClusterFinderConfig config;

    if (*padMajor)
    {
        config.layout = ChargemapLayout::PadMajor;
    }

    if (*tiling4x4)
    {
        config.layout = ChargemapLayout::Tiling4x4;
    }

    if (*tiling4x8)
    {
        config.layout = ChargemapLayout::Tiling4x8;
    }

    if (*tiling8x4)
    {
        config.layout = ChargemapLayout::Tiling8x4;
    }

    if (*scratchpad)
    {
        config.clusterbuilder = ClusterBuilder::ScratchPad;
    }

    config.halfPrecisionCharges = *halfs;
    config.splitCharges = *splitCharges;

    DataSet clusters;

    if (*cpu)
    {
        ReferenceClusterFinder cf(config);    

        std::vector<Cluster> res = cf.run(digits);
        clusters.serialize(res);
    }
    else
    {
        GPUClusterFinder cf;
        cf.setup(config, env, digits);
        auto cfRes = cf.run();
        clusters.serialize(cfRes.clusters);

        if (*peakFile)
        {
            auto peaks = cf.getPeaks();
            log::Info() << "Writing " << peaks.size()
                        << " peaks to file " << peakFile->Get();
            DataSet peakSet;
            peakSet.serialize<Digit>(peaks);
            peakSet.write(peakFile->Get());
        }
    }

    clusters.write(args::get(*clusterResultFile));


    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
