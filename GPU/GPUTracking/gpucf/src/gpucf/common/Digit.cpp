// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Digit.h"

#include <gpucf/common/RowInfo.h>
#include <gpucf/common/View.h>

#include <ostream>
#include <sstream>


using namespace gpucf;


static_assert(sizeof(Digit) == sizeof(PackedDigit), "");
static_assert(sizeof(PackedDigit) == PACKED_DIGIT_SIZE, "");


SectorMap<std::vector<Digit>> Digit::bySector(
        const SectorMap<std::vector<RawDigit>> &rawdigits)
{
    SectorMap<std::vector<Digit>> digits;  

    for (size_t i = 0; i < TPC_SECTORS; i++)
    {
        const std::vector<RawDigit> &raw = rawdigits[i];
        digits[i].reserve(raw.size());
        for (auto rd : raw)
        {
            digits[i].emplace_back(rd);
        }
    }

    return digits;
}


Digit::Digit()
    : Digit(0.f, 0, 0, 0)
{
}

Digit::Digit(const RawDigit &r)
    : Digit(r.charge, r.row, r.pad, r.time)
{
}

Digit::Digit(float _charge, int _row, int _pad, int _time)
{
    charge = static_cast<unsigned short>(_charge * 16.f) / 16.f;
    row = _row;
    pad = _pad;
    time = _time;
}


Object Digit::serialize() const
{
    Object obj("Digit");

    SET_FIELD(obj, row);
    SET_FIELD(obj, pad);
    SET_FIELD(obj, time);
    SET_FIELD(obj, charge);

    return obj;
}

void Digit::deserialize(const Object &o)
{
    /* GET_INT(o, cru); */
    GET_INT(o, row);
    GET_INT(o, pad);
    GET_INT(o, time);
    GET_FLOAT(o, charge);
}

int Digit::localRow() const
{
    return RowInfo::instance().globalToLocal(row);
}

int Digit::cru() const
{
    return RowInfo::instance().globalRowToCru(row);
}

bool Digit::operator==(const Digit &other) const
{
    bool eq = true;
    eq &= (row == other.row);
    eq &= (pad == other.pad);
    eq &= (time == other.time);
    eq &= (charge == other.charge);

    return eq;
}

std::ostream &gpucf::operator<<(std::ostream &os, const Digit &d) 
{
    return os << d.serialize();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
