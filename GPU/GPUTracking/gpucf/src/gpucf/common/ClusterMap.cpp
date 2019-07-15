#include "ClusterMap.h"

#include <limits>


using namespace gpucf;


void ClusterMap::add(const Cluster &c)
{
    clusters[c.globalRow()].push_back(c);
}

void ClusterMap::addAll(nonstd::span<const Cluster> cs)
{
    for (const Cluster &c : cs)
    {
        add(c);
    }
}

size_t ClusterMap::size() const
{
    size_t totalSize = 0;

    for (const auto &p : clusters)
    {
        totalSize += p.second.size();
    }

    return totalSize;
}

bool ClusterMap::contains(
        const Cluster &c) const
{
    auto tgtEntry = clusters.find(c.globalRow());

    if (tgtEntry == clusters.end())
    {
        return false;
    }

    const std::vector<Cluster> &tgtRow = tgtEntry->second;

    auto lookup = std::find_if(tgtRow.begin(), tgtRow.end(), 
            [&](const Cluster &c2) 
            { 
                return c.eq(c2, epsilonSmall, epsilonBig, mask); 
            });

    return lookup != tgtRow.end();
}

nonstd::optional<Cluster> ClusterMap::tryLookup(
        const Cluster &c) const
{
    auto tgtEntry = clusters.find(c.globalRow());

    if (tgtEntry == clusters.end())
    {
        return nonstd::nullopt;
    }

    const std::vector<Cluster> &tgtRow = tgtEntry->second;

    float dist = std::numeric_limits<float>::infinity();
    nonstd::optional<Cluster> lookup;

    for (const Cluster &other : tgtRow)
    {
        if (other.eq(c, epsilonSmall, epsilonBig, mask))
        {
            if (c.dist(other) < dist)
            {
                lookup = other;
            }
        }
    }

    return lookup;
}

std::vector<Cluster> ClusterMap::findDuplicates() const
{
    std::vector<Cluster> duplicates;

    for (const auto &it : clusters)
    {
        const std::vector<Cluster> &row = it.second;

        for (size_t i = 0; i < row.size(); i++)
        {
            const Cluster &curr = row[i];

            for (size_t j = 0; j < row.size(); j++)
            {
                if (i == j)
                {
                    continue;
                }

                if (curr.eq(row[j], epsilonSmall, epsilonBig, mask))
                {
                    duplicates.push_back(curr);     
                }
                
            }
        }
    }

    return duplicates;
}

void ClusterMap::setClusterEqParams(
        float epsilonSmall, 
        float epsilonBig,
        Cluster::FieldMask mask)
{
    this->epsilonSmall = epsilonSmall;
    this->epsilonBig   = epsilonBig;
    this->mask         = mask; 
}

// vim: set ts=4 sw=4 sts=4 expandtab:

