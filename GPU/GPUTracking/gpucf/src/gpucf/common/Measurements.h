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
