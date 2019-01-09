#pragma once

#include <iosfwd>


struct Digit 
{
    float charge;
    int cru;
    int row;
    int pad;
    int time;   

    Digit(float _charge, int _cru, int _row, int _pad, int _time)
        : charge(_charge)
        , cru(_cru)
        , row(_row)
        , pad(_pad)
        , time(_time)
    {
    }

};

std::ostream &operator<<(std::ostream &, const Digit &);

// vim: set ts=4 sw=4 sts=4 expandtab:
