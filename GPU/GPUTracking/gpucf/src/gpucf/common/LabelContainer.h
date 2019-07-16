#pragma once

#include <gpucf/common/MCLabel.h>
#include <gpucf/common/RawLabel.h>

#include <nonstd/span.hpp>

#include <vector>


namespace gpucf
{

class LabelContainer
{

public:

    LabelContainer(nonstd::span<const RawLabel>);

    nonstd::span<const MCLabel> operator[](size_t) const;

    size_t size() const;

private:

    std::vector<nonstd::span<const MCLabel>> viewById;
    std::vector<MCLabel> labels;

    void add(const RawLabel &);

};

} // namespace gpucf



// vim: set ts=4 sw=4 sts=4 expandtab:
