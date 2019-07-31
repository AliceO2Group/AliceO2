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
