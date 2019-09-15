// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <array>
#include <vector>


class RowInfo
{

public:
    static const RowInfo &instance();

    static constexpr int regionNum = 10;
    static const std::array<int, regionNum> rowsPerRegion;

    std::vector<int>              globalToLocalMap;
    std::vector<int>              globalRowToCruMap;
    std::vector<std::vector<int>> localToGlobalMap;

    RowInfo();

    int globalRowToCru(int) const;

    int globalToLocal(int) const;

    int localToGlobal(int, int) const;

    int numOfRows() const;
};

// vim: set ts=4 sw=4 sts=4 expandtab:
