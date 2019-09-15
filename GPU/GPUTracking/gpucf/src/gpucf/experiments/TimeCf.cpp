// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "TimeCf.h"

#include <gpucf/algorithms/ClusterFinderProfiler.h>
#include <gpucf/algorithms/ClusterFinderTest.h>
#include <gpucf/common/log.h>


using namespace gpucf;
namespace fs = filesystem;


TimeCf::TimeCf(
        const std::string &n,
        fs::path tgt,
        ClusterFinderConfig conf,
        nonstd::span<const Digit> d, 
        size_t N)
    : Experiment(conf)
    , name(n)
    , tgtFile(tgt)
    , repeats(N)
    , digits(d)
{
}

void TimeCf::run(ClEnv &env)
{

    log::Info() << "Benchmarking " << name << "(" << cfg << ")";

    Measurements measurements;

    for (size_t i = 0; i < repeats+1; i++)
    {
        ClEnv envCopy = env;

        ClusterFinderProfiler p(cfg, digits.size(), envCopy);
        auto res = p.run(digits);

        if (i > 0)
        {
            measurements.add(res);
            measurements.finishRun();
        }
    }

    save(tgtFile, measurements);
}
    

// vim: set ts=4 sw=4 sts=4 expandtab:
