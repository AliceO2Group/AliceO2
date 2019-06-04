#include "CfRunner.h"

#include <gpucf/common/DataSet.h>
#include <gpucf/common/log.h>
#include <gpucf/algorithms/GPUClusterFinder.h>


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
}

int CfRunner::mainImpl()
{
    ClEnv env(*envFlags); 

    DataSet digitSet;
    digitSet.read(args::get(*digitFile));

    std::vector<Digit> digits = digitSet.deserialize<Digit>();

    GPUClusterFinder::Config config;

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

    if (*halfs)
    {
        config.halfPrecisionCharges = true;
    }

    GPUClusterFinder cf;
    cf.setup(config, env, digits);
    auto cfRes = cf.run();

    DataSet clusters;
    clusters.serialize(cfRes.clusters);
    clusters.write(args::get(*clusterResultFile));

    if (*peakFile)
    {
        auto peaks = cf.getPeaks();
        log::Info() << "Writing " << peaks.size()
                    << " peaks to file " << peakFile->Get();
        DataSet peakSet;
        peakSet.serialize<Digit>(peaks);
        peakSet.write(peakFile->Get());
    }

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
