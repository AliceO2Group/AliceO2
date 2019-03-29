#include "CfRunner.h"

#include <gpucf/common/DataSet.h>
#include <gpucf/common/log.h>
#include <gpucf/gpu/GPUClusterFinder.h>


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
            required,
            "Cluster finder config (provide exactly one!)",
            args::Group::Validators::Xor);

    reference = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Use standard config",
            {"std"});

    chargemapIdxMacro = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Use macro for chargemap Idx",
            {"idxMacro"});

    tilingLayout = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Use tiling layout in charge map",
            {"tiling"});

    padMajor = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Use pad major in charge map",
            {"padMajor"});
}

int CfRunner::mainImpl()
{
    ClEnv env(*envFlags); 

    DataSet digitSet;
    digitSet.read(args::get(*digitFile));

    std::vector<Digit> digits = digitSet.deserialize<Digit>();

    GPUClusterFinder::Config config;

    if (*chargemapIdxMacro) 
    {
        config.useChargemapMacro = true;
    } 
    else if (*tilingLayout)
    {
        config.layout = ChargemapLayout::Tiling4x4;
    }
    else if (*padMajor)
    {
        config.layout = ChargemapLayout::PadMajor;
    }
    else if (!*reference)
    {
        log::Fail() << "Unknown configuration provided.";
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
