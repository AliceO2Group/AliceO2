#include "utils.h"

#include <gpucf/common/Digit.h>
#include <gpucf/common/LabelContainer.h>
#include <gpucf/common/log.h>

#include <TAxis.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TMultiGraph.h>


using namespace gpucf;


bool gpucf::peaksOverlap(
        const Digit &p1, 
        const Digit &p2, 
        const LabelContainer &c)
{
    View<MCLabel> labels1 = c[p1];
    View<MCLabel> labels2 = c[p2];

    size_t sharedLabels = 0;
    for (const MCLabel &l1 : labels1)
    {
        for (const MCLabel &l2 : labels2)
        {
            sharedLabels += (l1 == l2);
        }
    }

    return (sharedLabels >= 2);

    /* if (l1.size() != 2 || l2.size() != 2) */
    /* { */
    /*     return false; */
    /* } */
    /* else */
    /* { */
    /*     return (l1[0] == l2[0] && l1[1] == l2[1]) */ 
    /*         || (l1[0] == l2[1] && l1[1] == l2[0]); */
    /* } */

}


void gpucf::plot(
        const std::vector<std::string> &names,
        const std::vector<std::vector<int>> &data,
        const std::string &fname,
        const std::string &xlabel,
        const std::string &ylabel,
        const PlotConfig &cnf)
{
    size_t n = 0;
    for (auto &y : data)
    {
        n = std::max(n, y.size());
    }

    std::vector<int> x(n);
    for (size_t i = 0; i < x.size(); i++)
    {
        x[i] = i;
    }

    TCanvas *c = new TCanvas("c1", "Cluster per Track", 1200, 800);
    if (cnf.logYAxis)
    {
        c->SetLogy();
    }

    if (cnf.logXAxis)
    {
        c->SetLogx();
    }

    TMultiGraph *mg = new TMultiGraph();

    ASSERT(names.size() == data.size());
    for (size_t i = 0; i < names.size(); i++)
    {
        const std::vector<int> &y = data[i];
        TGraph *g = new TGraph(y.size(), x.data(), y.data());

        g->SetLineColor(i+2);
        g->SetMarkerColor(i+2);
        g->SetTitle(names[i].c_str());
        mg->Add(g);
    }

    mg->GetXaxis()->SetTitle(xlabel.c_str());
    mg->GetYaxis()->SetTitle(ylabel.c_str());
    mg->Draw(cnf.lineStyle.c_str());

    if (cnf.showLegend)
    {
        c->BuildLegend();
    }

    c->SaveAs(fname.c_str());
}

// vim: set ts=4 sw=4 sts=4 expandtab:

