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

#include <stdexcept>


class NoPlatformFoundError : public std::runtime_error
{
public:
    NoPlatformFoundError()
        : std::runtime_error("Could not find any OpenCL platform.")
    {
    }

};

class NoGpuFoundError : public std::runtime_error
{
public:
    NoGpuFoundError()
        : std::runtime_error("Could not find any GPU devices.")
    {
    }

};

class BuildFailedError : public std::runtime_error
{
public:
    BuildFailedError()
        : std::runtime_error("Failed to build all sources.")
    {
    }
    
};





// vim: set ts=4 sw=4 sts=4 expandtab:
