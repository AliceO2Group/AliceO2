#include "LabelContainer.h"

#include <gpucf/common/log.h>


using namespace gpucf;


SectorMap<LabelContainer> LabelContainer::bySector(
        const SectorData<RawLabel> &labels)
{
    SectorMap<LabelContainer> containers;

    size_t start = 0;
    for (size_t i = 0; i < TPC_SECTORS; i++)
    {
        size_t n = labels.elemsBySector[i];
        nonstd::span<const RawLabel> data(&labels.data[start], n);
        containers[i] = LabelContainer(data);
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

View<MCLabel> LabelContainer::operator[](size_t id) const
{
    return viewById.at(id);
}

size_t LabelContainer::size() const
{
    return viewById.size();
}

View<MCLabel> LabelContainer::allLabels() const
{
    return labels;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
