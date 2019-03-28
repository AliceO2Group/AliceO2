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
}

int CfRunner::mainImpl()
{
    ClEnv env(*envFlags); 

    DataSet digitSet;
    digitSet.read(args::get(*digitFile));

    std::vector<Digit> digits = digitSet.deserialize<Digit>();

    GPUClusterFinder::Config config;

    if (*reference)
    {
        config.usePackedDigits = true;
    }

    if (*chargemapIdxMacro)
    {
        config.usePackedDigits   = true;
        config.useChargemapMacro = true;
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
