// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Testbench.h"

#include <gpucf/debug/PeakCountTest.h>


using namespace gpucf;


Testbench::Testbench()
    : Executable("Run tests on GPU cluster finder.")
{
}

void Testbench::setupFlags(args::Group &required, args::Group &optional)
{
    envFlags  = std::make_unique<ClEnv::Flags>(required, optional); 
}

int Testbench::mainImpl()
{
    ClusterFinderConfig config;
    config.dbg = true;
    ClEnv env(*envFlags, config);
    PeakCountTest pctest(config, env);

    /* return pctest.run( */
    /*         {{7, 0, 0, 0, 7}, */
    /*          {0, 6, 0, 0, 0}, */
    /*          {0, 0, 5, 0, 0}, */
    /*          {0, 0, 0, 7, 0}, */
    /*          {7, 0, 0, 0, 9} */
    /*         }, */
    /*         {{0b11, 0, 0, 0, 0b11}, */
    /*          {0}, */
    /*          {0, 0, 0b10}, */
    /*          {0, 0, 0, 0b10}, */
    /*          {0b11, 0, 0, 0, 0b11} */
    /*         }, */
    /*         {{1, 0, 0, 0, 1}, */
    /*          {0}, */
    /*          {0, 0, -1}, */
    /*          {0, 0, 0, 1}, */
    /*          {1, 0, 0, 0, 1} */
    /*         } */
    /* ); */

    /* return pctest.run( */
    /*         {{7, 7, 1, 0, 0}, */
    /*          {0,15, 0, 0, 0}, */
    /*          {0,11,44, 3, 9}, */
    /*          {0, 0,16, 3, 2}, */
    /*          {0, 0, 0, 2, 0}}, */
    /*          {}, */
    /*          {} */
    /* ); */

    /* pctest.run( */
    /*         {{0}, */
    /*          {0, 0, 3, 0, 0}, */
    /*          {0, 0, 19, 5, 0}, */
    /*          {12, 14, 15, 12, 2}, */
    /*          {1, 3, 0, 12, 1}}, */
    /*          {}, */
    /*          {} */
    /* ); */

    /* log::Debug() << "=========================="; */

    /* pctest.run( */
    /*         {{0, 0, 3, 0, 0}, */
    /*          {0, 2, 3, 2, 0}, */
    /*          {0, 0, 18, 5, 0}, */
    /*          {0}, */
    /*          {0}}, */
    /*          {}, */
    /*          {} */
    /* ); */

    /* pctest.run( */
    /*         {{0}, */
    /*          {0}, */
    /*          {0, 56, 518, 79, 3}, */
    /*          {0, 0, 36, 2, 0, 0, 0, 0, 1}, */
    /*          {0, 28, 23, 1, 0, 0, 1} */
    /*         }, */
    /*         {}, */
    /*         {} */
    /* ); */
    /* pctest.run( */
    /*         {{0}, */
    /*          {0, 0, 518}, */
    /*          {0, 56, 518, 79, 3}, */
    /*          {0, 0, 36, 2, 0, 0, 0, 0, 1}, */
    /*          {0, 28, 23, 1, 0, 0, 1} */
    /*         }, */
    /*         {}, */
    /*         {} */
    /* ); */
    pctest.run(
            {{0, 9},
             {10, 37},
             {8, 90},
             {3, 81},
             {0, 28},
             {0, 4}
            },
            {},
            {}
    );

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:

