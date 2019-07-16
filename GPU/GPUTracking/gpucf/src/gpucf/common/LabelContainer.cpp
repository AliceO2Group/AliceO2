#include "LabelContainer.h"

#include <gpucf/common/log.h>


using namespace gpucf;


std::vector<LabelContainer> LabelContainer::bySector(
        const SectorData<RawLabel> &labels)
{
    std::vector<LabelContainer> containers;

    size_t start = 0;
    for (auto n : labels.elemsBySector)
    {
        nonstd::span<const RawLabel> data(&labels.data[start], n);
        containers.emplace_back(data);
        start += n;
    }

    return containers;
}

LabelContainer::LabelContainer(nonstd::span<const RawLabel> rawlabels)
{
    ASSERT(!rawlabels.empty());

    labels.reserve(rawlabels.size());

    ASSERT(rawlabels.front().id == 0);

    size_t start = 0;
    size_t elems = 1;
    int id = 0;

    labels.emplace_back(rawlabels.front());

    for (const RawLabel &l : 
            nonstd::span<const RawLabel>(&rawlabels[1], rawlabels.size()-1))
    {
        ASSERT(l.id == id || l.id == id+1);

        if (l.id == id+1)
        {
            viewById.emplace_back(&labels[start], elems); 

            start = labels.size();
            elems = 0;
            id++;
        }
        
        labels.emplace_back(l);
        elems++;
    }

    viewById.emplace_back(&labels[start], elems);
}

nonstd::span<const MCLabel> LabelContainer::operator[](size_t id) const
{
    return viewById.at(id);
}

size_t LabelContainer::size() const
{
    return viewById.size();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
