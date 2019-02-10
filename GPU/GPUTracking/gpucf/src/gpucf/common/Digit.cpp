#include "Digit.h"

#include <ostream>
#include <sstream>


using namespace gpucf;


static_assert(sizeof(PaddedDigit) == PADDED_DIGIT_SIZE);
static_assert(sizeof(Digit) == sizeof(PaddedDigit));
static_assert(sizeof(PackedDigit) == PACKED_DIGIT_SIZE);


Digit::Digit()
    : Digit(0, 0, 0, 0, 0)
{
}

Digit::Digit(float _charge, int _cru, int _row, int _pad, int _time)
{
    charge = _charge;
    cru = _cru;
    row = _row;
    pad = _pad;
    time = _time;
}

Object Digit::serialize() const
{
    Object obj("Digit");

    SET_FIELD(obj, cru);
    SET_FIELD(obj, row);
    SET_FIELD(obj, pad);
    SET_FIELD(obj, time);
    SET_FIELD(obj, charge);

    return obj;
}

void Digit::deserialize(const Object &o)
{
    GET_INT(o, cru);
    GET_INT(o, row);
    GET_INT(o, pad);
    GET_INT(o, time);
    GET_FLOAT(o, charge);
}

PackedDigit Digit::toPacked() const
{
    return PackedDigit{ charge, time, pad, row };
}

std::ostream &gpucf::operator<<(std::ostream &os, const Digit &d) 
{
    return os << d.serialize();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
