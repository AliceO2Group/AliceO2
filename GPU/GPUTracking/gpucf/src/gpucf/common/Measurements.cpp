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
        writer.write("start", step.start);
        writer.write("end", step.end);

        writer.close();
    }
};

template<>
struct default_value_writer<Measurement>
{
    void operator()(
            std::ostream &stream, 
            const Measurement &measurement,
            writer_configuration config)
    {
        minijson::object_writer writer(stream, config);

        writer.write("start", measurement.start);
        writer.write("end", measurement.end);

        {
            minijson::array_writer lanesWriter = writer.nested_array("lanes");
            /* const std::vector<Lane> &lanes = measurement.lanes; */
            for (const std::vector<Step> &lane : measurement.lanes)
            {
                lanesWriter.write_array(lane.begin(), lane.end());
            }

            lanesWriter.close();
        }

        writer.close();
    }
};

/* template<> */
/* struct default_value_writer<Measurements> */
/* { */
/*     void operator()( */ 
/*             std::ostream &stream, */
/*             const Measurements &measurement, */
/*             writer_configuration config) */
/*     { */
/*         minijson::object_writer writer(stream, config); */

/*         const std::vector<Measurement> &runs = measurement.getRuns(); */
/*         writer.write("runs", runs.begin(), runs.end()); */
/*     } */
/* }; */

} // namespace minijson


Step::Step(const std::string &myName, const Event &ev)
    : Step(myName, ev.startMs(), ev.endMs())
{
}

Step::Step(const std::string &myName, Timestamp s, Timestamp e)
    : name(myName)
    , start(s)
    , end(e)
{
}


void Measurements::add(const Measurement &m)
{
    runs.push_back(m);
}

const std::vector<Measurement> &Measurements::getRuns() const
{
    return runs;
}


std::ostream &gpucf::operator<<(std::ostream &os, const Measurements &m)
{
    minijson::object_writer writer(
            os,
            minijson::writer_configuration().pretty_printing(true));

    const std::vector<Measurement> &runs = m.getRuns();
    writer.write_array("runs", runs.begin(), runs.end());

    writer.close();

    return os;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
