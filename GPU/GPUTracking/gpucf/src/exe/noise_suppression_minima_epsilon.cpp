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
#include <gpucf/noisesuppression/NoiseSuppressionOverArea.h>
#include <gpucf/noisesuppression/utils.h>

#include <args/args.hxx>

#include <TAxis.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TMultiGraph.h>

#include <memory>
#include <vector>


using namespace gpucf;


struct HitnumPerPeaks
{
    int noPeaks = 0;
    int onePeak = 0;
    int twoPeaksOverlap = 0;
    int twoPeaksNoOverlap = 0;
    int tenPeaks = 0;
    int moreThanTenPeaks = 0;
};

std::unordered_map<TpcHitPos, std::vector<Digit>> sortPeaksByHit(
        const SectorMap<LabelContainer> &labels,
        const SectorMap<RowMap<std::vector<Digit>>> &peaks)
{
    std::unordered_map<TpcHitPos, std::vector<Digit>> hits;
    for (short sector = 0; sector < TPC_SECTORS; sector++)
    {
        for (short row = 0; row < TPC_NUM_OF_ROWS; row++)
        {
            for (const Digit &peak : peaks[sector][row])
            {
                for (const MCLabel &label : labels[sector][peak])
                {
                    hits[{sector, row, label}].push_back(peak);
                }
            }
        }
    }

    return hits;
}

std::unordered_set<TpcHitPos> findHits(
        const SectorMap<RowMap<std::vector<Digit>>> &peaks,
        const SectorMap<LabelContainer> &labels)
{
    std::unordered_set<TpcHitPos> hits;

    for (short sector = 0; sector < TPC_SECTORS; sector++)
    {
        for (short row = 0; row < TPC_NUM_OF_ROWS; row++)
        {
            for (const Digit &peak : peaks[sector][row])
            {
                for (const MCLabel &label : labels[sector][peak])
                {
                    hits.insert({sector, row, label});
                }
            }
        }
    }

    return hits;
}

std::vector<HitnumPerPeaks> sortHitsByPeaks(
        const std::vector<SectorMap<RowMap<std::vector<Digit>>>> &peaks,
        const SectorMap<RowMap<std::vector<Digit>>> &peaksGt,
        const SectorMap<LabelContainer> &labels)
{
    std::unordered_set<TpcHitPos> allHits = findHits(peaksGt, labels);

    std::vector<HitnumPerPeaks> hitsPerPeaks(peaks.size());
    for (size_t i = 0; i < peaks.size(); i++)
    {
        std::unordered_map<TpcHitPos, std::vector<Digit>> hits = 
            sortPeaksByHit(labels, peaks[i]);

        for (const TpcHitPos &hit : allHits)
        {
            auto hitWithPeaks = hits.find(hit);

            if (hitWithPeaks == hits.end())
            {
                hitsPerPeaks[i].noPeaks++;
            }
            else
            {
                const std::vector<Digit> &peaksOfHit = hitWithPeaks->second;

                size_t peaknum = peaksOfHit.size();

                if (peaknum == 2)
                {
                    bool overlap = peaksOverlap(
                            peaksOfHit[0],
                            peaksOfHit[1],
                            labels[hitWithPeaks->first.sector]);

                    hitsPerPeaks[i].twoPeaksOverlap   += overlap;
                    hitsPerPeaks[i].twoPeaksNoOverlap += !overlap;
                }
                else
                {
                    hitsPerPeaks[i].onePeak += (peaknum == 1);
                    hitsPerPeaks[i].tenPeaks += (peaknum > 2 && peaknum <= 10);
                    hitsPerPeaks[i].moreThanTenPeaks += (peaknum > 10);
                }
            }
        }
    }

    return hitsPerPeaks;
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


    std::vector<NoiseSuppressionOverArea> algorithms;
    for (size_t i = 0; i < 30; i++)
    {
        algorithms.emplace_back(2, 3, 3, i);
    }

    SectorMap<std::vector<Digit>> digits;
    {
        log::Info() << "Reading digit file " << args::get(digitfile);
        SectorMap<std::vector<RawDigit>> rawdigits = 
            gpucf::read<RawDigit>(args::get(digitfile));
        digits = Digit::bySector(rawdigits);
    }

    SectorMap<LabelContainer> labels;
    {
        log::Info() << "Reading label file " << args::get(labelfile);
        SectorMap<std::vector<RawLabel>> rawlabels = 
            gpucf::read<RawLabel>(args::get(labelfile));
        labels = LabelContainer::bySector(rawlabels, digits);
    }

    // map algorithm id -> result of algorithm
    std::vector<SectorMap<RowMap<std::vector<Digit>>>> filteredPeaks(
            algorithms.size());
    SectorMap<RowMap<std::vector<Digit>>> peaks;

    for (size_t sector = 0; sector < TPC_SECTORS; sector++)
    {
        log::Info() << "Processing sector " << sector << "...";
        Map<float> chargeMap(
                digits[sector], 
                [](const Digit &d) { return d.charge; }, 
                0.f);

        peaks[sector] = findPeaksByRow(digits[sector], chargeMap);

        RowMap<Map<bool>> peakmap = makePeakMapByRow(peaks[sector]);

        for (size_t id = 0; id < algorithms.size(); id++)
        {
            auto &alg = algorithms[id];
            filteredPeaks[id][sector] = 
                alg.run(peaks[sector], peakmap, chargeMap);
        }

    }
    
    std::vector<HitnumPerPeaks> epsilonToHits = 
            sortHitsByPeaks(filteredPeaks, peaks, labels);

    std::vector<std::string> names = {
        /* "no peaks", */
        /* "1 peak", */
        /* "2 peaks (overlap)", */
        /* "2 peaks (no overlap)", */
        /* "3 - 10 peaks", */
        /* "> 10 peaks" */
        "Keine Peaks",
        "1 Peak",
        "2 Peaks (mit Ueberlapp)",
        "2 Peaks (ohne Ueberlapp)",
        "3 - 10 Peaks",
        "> 10 Peaks",
    };

    std::vector<std::vector<int>> data(names.size());
    for (const HitnumPerPeaks &hitnumPerPeaks : epsilonToHits)
    {
        data[0].push_back(hitnumPerPeaks.noPeaks);
        data[1].push_back(hitnumPerPeaks.onePeak);
        data[2].push_back(hitnumPerPeaks.twoPeaksOverlap);
        data[3].push_back(hitnumPerPeaks.twoPeaksNoOverlap);
        data[4].push_back(hitnumPerPeaks.tenPeaks);
        data[5].push_back(hitnumPerPeaks.moreThanTenPeaks);
    }

    plot(names, data, "epsilonToHits.pdf", "epsilon", "# Hits");
}

// vim: set ts=4 sw=4 sts=4 expandtab:
