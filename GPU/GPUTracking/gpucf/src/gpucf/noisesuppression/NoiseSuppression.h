#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/common/Map.h>
#include <gpucf/common/RowMap.h>
#include <gpucf/common/View.h>

#include <vector>


namespace gpucf
{
    
class NoiseSuppression
{

public:

    RowMap<std::vector<Digit>> run(
            const RowMap<std::vector<Digit>> &,
            const RowMap<Map<bool>> &,
            const Map<float> &);

    std::vector<Digit> runOnAllRows(
            View<Digit>,
            const Map<bool> &,
            const Map<float> &);

    std::string getName() const
    {
        return name;
    }

protected:
    
    NoiseSuppression(const std::string &name)
        : name(name)
    {
    }

    virtual std::vector<Digit> runImpl(
            View<Digit>,
            const Map<bool> &,
            const Map<float> &) = 0;

private:

    std::string name;

};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

