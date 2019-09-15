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
#include <gpucf/executable/CfCLIFlags.h>
#include <gpucf/executable/Executable.h>


namespace gpucf
{

class CfRunner : public Executable
{
    
public:
    CfRunner();

protected:
    void setupFlags(args::Group &, args::Group &) override;
    int mainImpl() override;

private:
    std::unique_ptr<ClEnv::Flags> envFlags;
    std::unique_ptr<CfCLIFlags> cfflags;
    OptStringFlag digitFile;
    OptStringFlag clusterResultFile;
    OptStringFlag peakFile;

    OptFlag cpu;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

