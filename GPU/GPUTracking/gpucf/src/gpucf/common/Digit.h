#pragma once

#include <gpucf/common/Object.h>

#include <shared/Digit.h>

#include <iosfwd>
#include <string>


namespace gpucf
{

class Digit : public PaddedDigit
{
public:
    Digit();
    Digit(float, int, int, int, int);

    Object serialize() const;
    void deserialize(const Object &);

    PackedDigit toPacked() const;
};


std::ostream &operator<<(std::ostream &, const Digit &);

} // namespace gpucf


// vim: set ts=4 sw=4 sts=4 expandtab:
