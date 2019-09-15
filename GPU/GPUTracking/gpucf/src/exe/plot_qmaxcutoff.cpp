// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <gpucf/algorithms/cpu.h>
#include <gpucf/common/LabelContainer.h>
#include <gpucf/common/log.h>
#include <gpucf/common/RowMap.h>
#include <gpucf/common/SectorMap.h>
#include <gpucf/common/serialization.h>
#include <gpucf/common/TpcHitPos.h>

#include <args/args.hxx>

#include <TAxis.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TMultiGraph.h>

#include <vector>


using namespace gpucf;


struct PeakCount
{
    size_t peakNum0 = 0;
    size_t peakNum1 = 0;
    size_t peakNum2 = 0;
    size_t peakNum10 = 0;
    size_t peakNumX = 0;


    PeakCount() = default;

    PeakCount(const std::unordered_map<TpcHitPos, size_t> &peaksPerTrack)
    {
        for (auto p : peaksPerTrack)
        {
            update(p.second);
        }
    }

    PeakCount(const SectorMap<PeakCount> &peakCounts)
    {
        for (const PeakCount &pc : peakCounts)
        {
            peakNum0 += pc.peakNum0;
            peakNum1 += pc.peakNum1;
            peakNum2 += pc.peakNum2;
            peakNum10 += pc.peakNum10;
            peakNumX += pc.peakNumX;
        }
    }

    void update(size_t c)
    {
        peakNum0  += (c == 0);
        peakNum1  += (c == 1);
        peakNum2  += (c == 2);
        peakNum10 += (c > 2 && c <= 10);
        peakNumX  += (c > 10);
    }

    void update(const PeakCount &other)
    {
        peakNum0 += other.peakNum0;
        peakNum1 += other.peakNum1;
        peakNum2 += other.peakNum2;
        peakNum10 += other.peakNum10;
        peakNumX += other.peakNumX;
    }

};

std::ostream &operator<<(std::ostream &o, const PeakCount &pc)
{
    return o << "Number of tracks with...\n"
             << " ... no peaks   -> " << pc.peakNum0 << "\n"
             << " ... one peak   -> " << pc.peakNum1 << "\n"
             << " ... two peaks  -> " << pc.peakNum2 << "\n"
             << " ... 3-10 peaks -> " << pc.peakNum10 << "\n"
             << " ... > 10 peaks -> " << pc.peakNumX;
}


PeakCount countPeaksPerHitWithCutoff(
        View<Digit> peaks,
        const LabelContainer &labels,
        size_t cutoff)
{
    std::unordered_map<TpcHitPos, size_t> peaksPerHit;

    for (const Digit &peak : peaks)
    {
        for (const MCLabel &label : labels[peak])
        {
            peaksPerHit[{0, peak.row, label}] += (peak.charge >= cutoff);
        }
    }

    return PeakCount(peaksPerHit);
}

std::vector<PeakCount> peaksPerHitCutoff(
        const SectorMap<std::vector<Digit>> &digits,
        const SectorMap<Map<float>> &chargeMaps,
        const SectorMap<LabelContainer> &labels,
        size_t maxCutoff)
{
    std::vector<PeakCount> peakcounts(maxCutoff+1);    

    for (short sector = 0; sector < TPC_SECTORS; sector++)
    {
        log::Info() << "Processing sector " << sector;

        std::vector<Digit> peaks = findPeaks(digits[sector], chargeMaps[sector]);

        for (size_t cutoff = 0; cutoff <= maxCutoff; cutoff++)
        {
            peakcounts[cutoff].update(
                    countPeaksPerHitWithCutoff(peaks, labels[sector], cutoff));
        }
    }

    return peakcounts;
}
        

void plotPeakCounts(View<PeakCount> counts, const std::string &file)
{
    const size_t n = counts.size();
    std::vector<int> x(n);
    std::vector<int> pc0(n);
    std::vector<int> pc1(n);
    /* std::vector<int> pc2(n); */
    /* std::vector<int> pc10(n); */
    std::vector<int> pcX(n);

    for (size_t i = 0; i < n; i++)
    {
        x[i] = i;

        const PeakCount &pc = counts[i];
        pc0[i] = pc.peakNum0;
        pc1[i] = pc.peakNum1;
        /* pc2[i] = pc.peakNum2; */
        /* pc10[i] = pc.peakNum10; */
        pcX[i] = pc.peakNum2 + pc.peakNum10 + pc.peakNumX;
    }

    TCanvas *c = new TCanvas("c1", "Cluster per Track", 1200, 800);
    TMultiGraph *mg = new TMultiGraph();

    TGraph *g0 = new TGraph(n, x.data(), pc0.data());
    g0->SetLineColor(1);
    g0->SetMarkerColor(1);
    g0->SetTitle("Hits ohne Peak");
    mg->Add(g0);

    TGraph *g1 = new TGraph(n, x.data(), pc1.data());
    g1->SetLineColor(3);
    g1->SetMarkerColor(3);
    g1->SetTitle("Hits mit einem Peak");
    mg->Add(g1);

    /* TGraph *g2 = new TGraph(n, x.data(), pc2.data()); */
    /* g2->SetLineColor(3); */
    /* g2->SetMarkerColor(3); */
    /* g2->SetTitle("# Hits w. 2 peak"); */
    /* mg->Add(g2); */

    /* TGraph *g10 = new TGraph(n, x.data(), pc10.data()); */
    /* g10->SetLineColor(4); */
    /* g10->SetMarkerColor(4); */
    /* g10->SetTitle("# Hits w. 3-10 peaks"); */
    /* mg->Add(g10); */

    TGraph *gX = new TGraph(n, x.data(), pcX.data());
    gX->SetLineColor(2);
    gX->SetMarkerColor(2);
    gX->SetTitle("Hits mit >1 Peaks");
    mg->Add(gX);

    mg->Draw("AL");
    mg->GetXaxis()->SetTitle("Schwellenwert qmax");
    mg->GetYaxis()->SetTitle("# Hits");
    c->BuildLegend();
    c->SaveAs(file.c_str());
}


int main(int argc, const char *argv[])
{
    args::ArgumentParser parser("");

    args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});

    args::ValueFlag<std::string> digitfile(parser, "D", "Digit file", {'d', "digits"});
    args::ValueFlag<std::string> labelfile(parser, "L", "Label file", {'l', "labels"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Help &)
    {
        std::cerr << parser;
        std::exit(1);
    }

    log::Info() << "Reading digit file " << args::get(digitfile);
    SectorMap<std::vector<RawDigit>> rawdigits = 
            gpucf::read<RawDigit>(args::get(digitfile));
    SectorMap<std::vector<Digit>> digits = Digit::bySector(rawdigits);

    log::Info() << "Reading label file " << args::get(labelfile);
    SectorMap<std::vector<RawLabel>> rawlabels = 
            gpucf::read<RawLabel>(args::get(labelfile));
    SectorMap<LabelContainer> labels = 
            LabelContainer::bySector(rawlabels, digits);

    log::Info() << "Creating chargemap";
    SectorMap<Map<float>> chargemaps;
    for (size_t sector = 0; sector < TPC_SECTORS; sector++)
    {
        chargemaps[sector] = Map<float>(
                digits[sector], 
                [](const Digit &d) { return d.charge; }, 
                0.f);
    }

    std::vector<PeakCount> cutoffToPeakCount = 
            peaksPerHitCutoff(digits, chargemaps, labels, 30);

    for (size_t cutoff = 0; cutoff < cutoffToPeakCount.size(); cutoff++)
    {
        log::Info() << "*** cutoff = " << cutoff;
        log::Info() << cutoffToPeakCount[cutoff];
    }

    plotPeakCounts(cutoffToPeakCount, "cutoffToPeakCount.pdf");

    return 0;

}

// vim: set ts=4 sw=4 sts=4 expandtab:
