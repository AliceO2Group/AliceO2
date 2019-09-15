// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Position.h"


using namespace gpucf;


Position::Position(const Digit &d)
    : Position(d, 0, 0)
{
}

Position::Position(const Digit &d, int dp, int dt)
    : Position(d.row, d.pad + dp, d.time + dt)
{
}

Position::Position(row_t r, pad_t p, timestamp t)
    : row(r)
    , pad(p)
    , time(t)
{
}

bool Position::operator==(const Position &other) const
{
    return idx() == other.idx();
}

size_t Position::idx() const
{
    return TPC_NUM_OF_PADS * time + tpcGlobalPadIdx(row, pad);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
