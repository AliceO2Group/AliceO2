// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Experiment.h"

#include <gpucf/common/log.h>

#include <fstream>


using namespace gpucf;
namespace fs = filesystem;


Experiment::Experiment(ClusterFinderConfig cfg)
    : cfg(cfg)
{
}

Experiment::~Experiment()
{
}

void Experiment::save(fs::path fname, const Measurements &data)
{
    log::Info() << "Writing measurements to " << fname;

    std::ofstream out(fname.str());
    out << data;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
