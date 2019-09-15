// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/common/ClEnv.h>
#include <gpucf/common/Measurements.h>

#include <filesystem/path.h>


namespace gpucf
{

class Experiment
{
public:
    Experiment(ClusterFinderConfig cfg);

    virtual ~Experiment();

    virtual void run(ClEnv &) = 0;

    ClusterFinderConfig getConfig() const 
    {
        return cfg;
    }

protected:
    ClusterFinderConfig cfg;

    void save(filesystem::path, const Measurements &);

};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
