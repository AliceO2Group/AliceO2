#include "CfRunner.h"

#include <gpucf/algorithms/ClusterFinderTest.h>
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

    cpu = INIT_FLAG(
            args::Flag,
            *cfconfig,
            "",
            "Run cluster finder on cpu.",
            {"cpu"});

    #define CLUSTER_FINDER_FLAG(name, val, def, desc) \
            name = INIT_FLAG( \
                    args::Flag, \
                    *cfconfig, \
                    "", \
                    desc, \
                    {#name});
    #include <gpucf/algorithms/ClusterFinderFlags.def>

    #define MEMORY_LAYOUT(name, def, desc) \
            layout##name = INIT_FLAG( \
                    args::Flag, \
                    *cfconfig, \
                    "", \
                    desc, \
                    {"layout" #name});
    #include <gpucf/algorithms/ClusterFinderFlags.def>

    #define CLUSTER_BUILDER(name, def, desc) \
            builder##name = INIT_FLAG( \
                    args::Flag, \
                    *cfconfig, \
                    "", \
                    desc, \
                    {"builder" #name});
    #include <gpucf/algorithms/ClusterFinderFlags.def>
}

int CfRunner::mainImpl()
{

    DataSet digitSet;
    digitSet.read(args::get(*digitFile));

    std::vector<Digit> digits = digitSet.deserialize<Digit>();

    ClusterFinderConfig config;

    #define CLUSTER_FINDER_FLAG(name, val, def, desc) config.name = *name;
    #include <gpucf/algorithms/ClusterFinderFlags.def>

    #define MEMORY_LAYOUT(name, def, desc) \
        if (*layout##name) \
        { \
            config.layout = ChargemapLayout::name; \
        }
    #include <gpucf/algorithms/ClusterFinderFlags.def>

    #define CLUSTER_BUILDER(name, def, desc) \
        if (*builder##name) \
        { \
            config.clusterbuilder = ClusterBuilder::name; \
        }
    #include <gpucf/algorithms/ClusterFinderFlags.def>


    DataSet clusters;

    if (*cpu)
    {
        ReferenceClusterFinder cf(config);    

        auto res = cf.run(digits);
        clusters.serialize(res.cluster);
    }
    else
    {
#if 0
        GPUClusterFinder cf;
        ClEnv env(*envFlags, config); 
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
#else
        ClEnv env(*envFlags, config); 
        ClusterFinderTest cf(config, digits.size(), env);
        cf.run(digits);
#endif
    }

#if 0
    clusters.write(args::get(*clusterResultFile));
#endif

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
