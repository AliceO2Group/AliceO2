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

#include <string>
#include <vector>


namespace gpucf
{

class Digit;
class LabelContainer;


struct PlotConfig
{
    std::string lineStyle = "A*";
    bool showLegend = true;
    bool logXAxis = false;
    bool logYAxis = false;
};

void plot(
        const std::vector<std::string> &names,
        const std::vector<std::vector<int>> &vals,
        const std::string &fname,
        const std::string &xlabel,
        const std::string &ylabel,
        const PlotConfig &cnf={});


bool peaksOverlap(const Digit &, const Digit &, const LabelContainer &);
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
