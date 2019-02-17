#pragma once

#include <gpucf/common/Timestamp.h>

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>


namespace gpucf
{
    class Event;

    struct Step
    {
        Step(const std::string &, const Event &);
        Step(const std::string &, Timestamp, Timestamp);

        std::string name;
        Timestamp start;
        Timestamp end;
    };

    using Lane = std::vector<Step>;

    struct Measurement
    {
        Timestamp start;
        Timestamp end;
        std::vector<Lane> lanes; 
    };

    class Measurements
    {
    public:
        void add(const Measurement &);

        const std::vector<Measurement> &getRuns() const;

    private:
        std::vector<Measurement> runs;
    };

    std::ostream &operator<<(std::ostream &, const Measurements &);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
