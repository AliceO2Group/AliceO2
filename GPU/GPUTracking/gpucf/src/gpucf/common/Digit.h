#pragma once

#include <gpucf/common/Object.h>

#include <shared/Digit.h>

#include <iosfwd>
#include <string>


namespace gpucf
{

class Digit : public PackedDigit
{
public:
    Digit();
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
