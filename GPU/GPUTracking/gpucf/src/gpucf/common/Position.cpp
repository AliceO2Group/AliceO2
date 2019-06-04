#include "Position.h"


using namespace gpucf;


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
