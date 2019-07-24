#include "LabelContainer.h"

#include <gpucf/common/log.h>

#include <unordered_set>


using namespace gpucf;


SectorMap<LabelContainer> LabelContainer::bySector(
        const SectorMap<std::vector<RawLabel>> &labels,
        const SectorMap<std::vector<Digit>> &digits)
{
    SectorMap<LabelContainer> containers;

    for (size_t i = 0; i < TPC_SECTORS; i++)
    {
        containers[i] = LabelContainer(labels[i], digits[i]);
    }

    return containers;
}

LabelContainer::LabelContainer(View<RawLabel> rawlabels, View<Digit> digits)
{
    ASSERT(!rawlabels.empty());

    labels.reserve(rawlabels.size());

    ASSERT(rawlabels.front().id == 0);

    size_t start = 0;
    size_t elems = 1;
    int id = 0;

    labels.emplace_back(rawlabels.front());

    viewById.reserve(digits.size());

    for (const RawLabel &l : View<RawLabel>(&rawlabels[1], rawlabels.size()-1))
    {
        ASSERT(l.id == id || l.id == id+1);

        if (l.id == id+1)
        {
            View<MCLabel> view(&labels[start], elems);
            viewById.push_back(view); 
            viewByPosition[digits[id]] = view;

            start = labels.size();
            elems = 0;
            id++;
        }
        
        labels.emplace_back(l);
        elems++;
    }

    View<MCLabel> view(&labels[start], elems);
    viewById.push_back(view); 
    viewByPosition[digits[id]] = view;
}

View<MCLabel> LabelContainer::operator[](const Position &p) const
{
    return viewByPosition.at(p);
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

size_t LabelContainer::countTracks() const
{
    std::unordered_set<MCLabel> uniqueTracks;

    for (const MCLabel &label : labels)
    {
        uniqueTracks.insert(label);
    }

    return uniqueTracks.size();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
