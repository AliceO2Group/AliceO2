#include "Digit.h"

#include <ostream>
#include <sstream>


using namespace gpucf;


static_assert(sizeof(FloatDigit) == FLOAT_DIGIT_SIZE);
static_assert(sizeof(Digit) == sizeof(FloatDigit));
static_assert(sizeof(HalfDigit) == HALF_DIGIT_SIZE);


Digit::Digit()
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

std::ostream &gpucf::operator<<(std::ostream &os, const Digit &d) 
{
    return os << d.serialize();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
