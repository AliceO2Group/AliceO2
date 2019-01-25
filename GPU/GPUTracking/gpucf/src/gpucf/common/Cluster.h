#pragma once

#include <gpucf/Object.h>

#include <shared/Cluster.h>

#include <iosfwd>
#include <string>


namespace gpucf
{

class Cluster : public FloatCluster
{

public:
    Cluster();
    Cluster(int, int, float, float, float, float, float, float);

    Object serialize() const;
    void deserialize(const Object &);

    bool hasNaN() const;
    bool hasNegativeEntries() const;

    bool operator==(const Cluster &) const;
};

std::ostream &operator<<(std::ostream &, const Cluster &);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
