#include "CfRunner.h"

#include <gpucf/algorithms/ClusterFinderTest.h>
#include <gpucf/algorithms/ReferenceClusterFinder.h>
#include <gpucf/common/DataSet.h>
#include <gpucf/common/RawLabel.h>
#include <gpucf/common/log.h>
#include <gpucf/common/serialization.h>


using namespace gpucf;


CfRunner::CfRunner()
    : Executable("Runs the GPU cluster finder.")
{
}

void CfRunner::setupFlags(args::Group &required, args::Group &optional)
{
    envFlags = std::make_unique<ClEnv::Flags>(required, optional); 
    cfflags = std::make_unique<CfCLIFlags>(required, optional);

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

    cpu = INIT_FLAG(
            args::Flag,
            optional,
            "",
            "Run cluster finder on cpu.",
            {"cpu"});

}

int CfRunner::mainImpl()
{
    SectorMap<std::vector<RawDigit>> rawdigits = read<RawDigit>(args::get(*digitFile));
    SectorMap<std::vector<Digit>> alldigits = Digit::bySector(rawdigits);

    std::vector<Digit> &digits = alldigits[0];


    /* std::vector<Digit> digits = read<RawDigit, Digit>(args::get(*digitFile)); */

    /* DataSet digitSet; */
    /* digitSet.read(args::get(*digitFile)); */

    /* std::vector<Digit> digits = digitSet.deserialize<Digit>(); */

    ClusterFinderConfig config = cfflags->asConfig();

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
