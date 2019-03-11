#include "Measurements.h"

#include <gpucf/common/Event.h>

#include <minijson_writer.hpp>


using namespace gpucf;


namespace minijson
{
    
template<>
struct default_value_writer<Step>
{
    void operator()(
            std::ostream &stream, 
            const Step &step,
            writer_configuration config)
    {
        minijson::object_writer writer(stream, config);

        writer.write("name", step.name);
        writer.write("queued", step.queued);
        writer.write("submitted", step.submitted);
        writer.write("start", step.start);
        writer.write("end", step.end);
        writer.write("lane", step.lane);
        writer.write("run", step.run);

        writer.close();
    }
};

} // namespace minijson


Step::Step(const std::string &name, const Event &ev)
    : Step(name, ev.queued(), ev.submitted(), ev.start(), ev.end())
{
}

Step::Step(const std::string &name, 
           Timestamp queued, 
           Timestamp submitted,
           Timestamp start,
           Timestamp end)
    : name(name)
    , queued(queued)
    , submitted(submitted)
    , start(start)
    , end(end)
{
}


void Measurements::add(nonstd::span<const Step> steps)
{
    for (const Step &step : steps)
    {
        add(step);
    }
}

void Measurements::add(const Step &step)
{
    steps.push_back(step);
    steps.back().run = run;
}

const std::vector<Step> &Measurements::getSteps() const
{
    return steps;
}

void Measurements::finishRun()
{
    run++;
}


std::ostream &gpucf::operator<<(std::ostream &os, const Measurements &m)
{
    minijson::object_writer writer(
            os,
            minijson::writer_configuration().pretty_printing(true));

    const std::vector<Step> &steps = m.getSteps();
    writer.write_array("steps", steps.begin(), steps.end());

    writer.close();

    return os;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
