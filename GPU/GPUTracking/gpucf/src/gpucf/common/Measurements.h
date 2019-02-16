#pragma once

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

        std::string name;
        float start;
        float end;
    };

    using Lane = std::vector<Step>;

    struct Measurement
    {
        float start;
        float end;
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
