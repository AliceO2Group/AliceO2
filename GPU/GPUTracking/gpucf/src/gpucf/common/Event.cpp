// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Event.h"


using namespace gpucf;


Event::Event()
{
}

cl::Event *Event::get()
{
    return &event;
}

Timestamp Event::queued() const
{
    return profilingInfo(CL_PROFILING_COMMAND_QUEUED);
}

Timestamp Event::submitted() const
{
    return profilingInfo(CL_PROFILING_COMMAND_SUBMIT);
}

Timestamp Event::start() const
{
    return profilingInfo(CL_PROFILING_COMMAND_START);
}

Timestamp Event::end() const
{
    return profilingInfo(CL_PROFILING_COMMAND_END);
}

Timestamp Event::profilingInfo(cl_profiling_info key) const
{
    Timestamp data; 
    event.getProfilingInfo(key, &data);

    return data;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
