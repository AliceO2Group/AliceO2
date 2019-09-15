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

#include <gpucf/common/Event.h>

#include <CL/cl2.hpp>


namespace gpucf
{

class Kernel1D
{

public:
    Kernel1D(const std::string &, cl::Program);

    std::string getName() const
    { 
        return name;
    }

    const Event &getEvent() const
    {
        return event;
    }

    template<typename T>
    void setArg(size_t n, T arg)
    {
        kernel.setArg(n, arg);
    }

    void call(size_t, size_t, size_t, cl::CommandQueue);

protected:
    cl::Kernel kernel;


private:
    std::string name;
    Event event;
    
};

} // namespace gpucf

#define DECL_KERNEL(type, name) type(cl::Program prg) : Kernel1D(name, prg) {}

// vim: set ts=4 sw=4 sts=4 expandtab:
