#pragma once

#include <gpucf/common/Object.h>
#include <gpucf/common/RawDigit.h>
#include <gpucf/common/SectorMap.h>
#include <gpucf/common/serialization.h>

#include <shared/Digit.h>

#include <iosfwd>
#include <string>
#include <vector>


namespace gpucf
{

class Digit : public PackedDigit
{

public:

    static SectorMap<std::vector<Digit>> bySector(const SectorData<RawDigit> &);

    Digit();
    Digit(const RawDigit &);
    Digit(float, int, int, int);

    Object serialize() const;
    void deserialize(const Object &);

    int localRow() const;
    int cru() const;

    bool operator==(const Digit &) const;
};

std::ostream &operator<<(std::ostream &, const Digit &);

} // namespace gpucf


// vim: set ts=4 sw=4 sts=4 expandtab:
