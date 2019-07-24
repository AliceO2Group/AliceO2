#pragma once

#include <gpucf/common/MCLabel.h>
#include <gpucf/common/Position.h>
#include <gpucf/common/RawLabel.h>
#include <gpucf/common/SectorMap.h>
#include <gpucf/common/serialization.h>
#include <gpucf/common/View.h>

#include <nonstd/span.hpp>

#include <vector>


namespace gpucf
{

class LabelContainer
{

public:

    static SectorMap<LabelContainer> bySector(
            const SectorMap<std::vector<RawLabel>> &,
            const SectorMap<std::vector<Digit>> &);

    LabelContainer() = default;
    LabelContainer(View<RawLabel>, View<Digit>);

    View<MCLabel> operator[](size_t) const;
    View<MCLabel> operator[](const Position &) const;

    size_t size() const;

    View<MCLabel> allLabels() const;

    size_t countTracks() const;

private:

    std::unordered_map<Position, View<MCLabel>> viewByPosition;
    std::vector<View<MCLabel>> viewById;
    std::vector<MCLabel> labels;

    void add(const RawLabel &);

};

} // namespace gpucf



// vim: set ts=4 sw=4 sts=4 expandtab:
