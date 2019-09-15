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

#include <gpucf/common/Timestamp.h>

#include <nonstd/span.hpp>

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>


namespace gpucf
{

class Event;
class Kernel1D;

struct Step
{

    Step(const Kernel1D &);

    Step(const std::string &, const Event &);

    Step(const std::string &, Timestamp, Timestamp, Timestamp, Timestamp);

    std::string name;
    Timestamp queued;
    Timestamp submitted;
    Timestamp start;
    Timestamp end;
    
    size_t lane = 0;
    size_t run  = 0;
};

class Measurements
{
public:
    void add(nonstd::span<const Step>);
    void add(const Step &);

    void finishRun();

    const std::vector<Step> &getSteps() const;

private:
    std::vector<Step> steps;

    size_t run = 0;
};

std::ostream &operator<<(std::ostream &, const Measurements &);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
