#include <gpucf/algorithms/cpu.h>
#include <gpucf/common/LabelContainer.h>
#include <gpucf/common/log.h>
#include <gpucf/common/RowMap.h>
#include <gpucf/common/SectorMap.h>
#include <gpucf/common/serialization.h>

#include <args/args.hxx>

#include <TCanvas.h>
#include <TGraph.h>
#include <TMultiGraph.h>

#include <memory>
#include <vector>


using namespace gpucf;


class NoiseSuppression
{

public:

    RowMap<std::vector<Digit>> run(
            const RowMap<std::vector<Digit>> &digits,
            const RowMap<Map<bool>> &isPeak,
            const Map<float> &chargemap)
    {
        RowMap<std::vector<Digit>> filteredPeaks;

        for (size_t row = 0; row < TPC_NUM_OF_ROWS; row++)
        {
            filteredPeaks[row] = runImpl(digits[row], isPeak[row], chargemap);
        }

        return filteredPeaks;
    }

    std::string getName() const
    {
        return name;
    }

protected:
    
    NoiseSuppression(const std::string &name)
        : name(name)
    {
    }

    virtual std::vector<Digit> runImpl(
            View<Digit>,
            const Map<bool> &,
            const Map<float> &) = 0;

private:

    std::string name;

};


class NoNoiseSuppression : public NoiseSuppression
{

public:

    NoNoiseSuppression() : NoiseSuppression("unfiltered")
    {
    }

protected:

    std::vector<Digit> runImpl(
            View<Digit> digits, 
            const Map<bool> &, 
            const Map<float> &)
    {
        return std::vector<Digit>(digits.begin(), digits.end());
    }
    
};


class QmaxCutoff : public NoiseSuppression
{
    
public:

    QmaxCutoff() : NoiseSuppression("qmax cutoff")
    {
    }

protected:

    std::vector<Digit> runImpl(
            View<Digit> digits,
            const Map<bool> &,
            const Map<float> &)
    {
        std::vector<Digit> filtered;
        for (const Digit &d : digits)
        {
            if (d.charge > 2)
            {
                filtered.push_back(d);
            }
        }

        return filtered;
    }

};


class NoiseSuppression5x5 : public NoiseSuppression
{

public:

    NoiseSuppression5x5() : NoiseSuppression("5x5 noise suppression")
    {
    }

protected:

    std::vector<Digit> runImpl(
            View<Digit> peaks,
            const Map<bool> &peakMap,
            const Map<float> &chargeMap)
    {
        std::vector<Digit> filtered;

        for (const Digit &p : peaks)
        {

            if (p.charge <= 2)
            {
                continue;
            }
            
            bool removeMe = false;

            for (int dp = -2; dp <= 2; dp++)
            {
                for (int dt = -2; dt <= 2; dt++)
                {
                    if (std::abs(dp) < 2 && std::abs(dt) < 2)
                    {
                        continue;
                    }

                    Position other(p, dp, dt);
                    /* Position between(digits[i], clamp(dp, -1, 1), clamp(dt, -1, 1)); */

                    float q = p.charge;
                    float oq = chargeMap[other];
                    /* float bq = chargemap[between]; */

                    bool otherIsPeak = peakMap[other];

                    removeMe |= otherIsPeak && (oq > q); //&& (q - bq <= 2);
                }
            }

            if (!removeMe)
            {
                filtered.push_back(p);
            }

        }

        return filtered;
    }
    
};


RowMap<Map<bool>> makePeakMapByRow(const RowMap<std::vector<Digit>> &peaks)
{
    RowMap<Map<bool>> peakMaps;

    for (size_t row = 0; row < peaks.size(); row++)
    {
        peakMaps[row] = Map<bool>(peaks[row], true, false);
    }

    return peakMaps;
}


struct TpcHitPos
{
    short sector;
    short row;
    MCLabel label;

    bool operator==(const TpcHitPos &other) const
    {
        return sector == other.sector 
            && row == other.row 
            && label == other.label;
    }
};

namespace std
{
    
    template<>
    struct hash<TpcHitPos>
    {
        size_t operator()(const TpcHitPos &p) const
        {
            static_assert(sizeof(p.label.event) == sizeof(short));
            static_assert(sizeof(p.label.track) == sizeof(short));

            size_t h = (size_t(p.sector)      << (8*3*sizeof(short)))
                     | (size_t(p.row)         << (8*2*sizeof(short)))
                     | (size_t(p.label.event) << (8*1*sizeof(short)))
                     | size_t(p.label.track);
            return std::hash<size_t>()(h);
        } 
    };

} // namespace std

std::vector<int> countPeaksPerTrack(
        const SectorMap<RowMap<std::vector<Digit>>> &peaks, 
        const SectorMap<LabelContainer> &labels)
{
    std::unordered_map<TpcHitPos, int> trackToPeaknum;    

    for (short sector = 0; sector < TPC_SECTORS; sector++)
    {
        for (short row = 0; row < TPC_NUM_OF_ROWS; row++)
        {
            for (const Digit &p : peaks[sector][row])
            {
                for (const MCLabel &label : labels[sector][p])
                {
                    trackToPeaknum[{sector, row, label}]++;
                }
            }
        }
    }

    int maxPeaks = 0;
    for (auto &p : trackToPeaknum)
    {
        maxPeaks = std::max(maxPeaks, p.second);
    }

    std::vector<int> peaknumToTracknum(maxPeaks+1);
    for (auto &p : trackToPeaknum)
    {
        peaknumToTracknum[p.second]++;
    }

    return peaknumToTracknum;
}

void plotPeaknumToTracknum(
        const std::vector<std::string> &names,
        const std::vector<std::vector<int>> &peaknumToTracknum,
        const std::string &fname)
{
    size_t n = 0;
    for (auto &vals : peaknumToTracknum)
    {
        n = std::max(n, vals.size());
    }

    std::vector<int> x(n);
    for (size_t i = 0; i < x.size(); i++)
    {
        x[i] = i;
    }

    TCanvas *c = new TCanvas("c1", "Cluster per Track", 1200, 800);
    c->SetLogy();
    c->SetLogx();
    TMultiGraph *mg = new TMultiGraph();

    ASSERT(names.size() == peaknumToTracknum.size());
    for (size_t i = 0; i < names.size(); i++)
    {
        const std::vector<int> &y = peaknumToTracknum[i];
        TGraph *g = new TGraph(y.size(), x.data(), y.data());
        g->SetLineColor(i+2);
        g->SetMarkerColor(i+2);
        g->SetTitle(names[i].c_str());
        mg->Add(g);
    }

    mg->Draw("A*");
    c->BuildLegend();
    c->SaveAs(fname.c_str());
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

    SectorMap<Map<float>> chargemaps;
    for (size_t sector = 0; sector < TPC_SECTORS; sector++)
    {
        chargemaps[sector] = Map<float>(
                digits[sector], 
                [](const Digit &d) { return d.charge; }, 
                0.f);
    }


    std::vector<std::unique_ptr<NoiseSuppression>> noiseSuppressionAlgos;
    noiseSuppressionAlgos.emplace_back(new NoNoiseSuppression);
    noiseSuppressionAlgos.emplace_back(new QmaxCutoff);
    noiseSuppressionAlgos.emplace_back(new NoiseSuppression5x5);


    // map algorithm id -> result of algorithm
    std::vector<SectorMap<RowMap<std::vector<Digit>>>> filteredPeaks(
            noiseSuppressionAlgos.size());
    for (size_t sector = 0; sector < TPC_SECTORS; sector++)
    {
        RowMap<std::vector<Digit>> peaks = 
                findPeaksByRow(digits[sector], chargemaps[sector]);

        RowMap<Map<bool>> peakmap = makePeakMapByRow(peaks);

        for (size_t id = 0; id < noiseSuppressionAlgos.size(); id++)
        {
            auto &algo = noiseSuppressionAlgos[id];
            filteredPeaks[id][sector] = 
                    algo->run(peaks, peakmap, chargemaps[sector]);
        }
    }

    // map algorithm id, N -> num of tracks with N peaks (in a row)
    std::vector<std::vector<int>> peaknumToTracknum(
            noiseSuppressionAlgos.size());
    for (size_t id = 0; id < noiseSuppressionAlgos.size(); id++)
    {
        peaknumToTracknum[id] = countPeaksPerTrack(filteredPeaks[id], labels);
    }


    std::vector<std::string> names;
    for (auto &algo : noiseSuppressionAlgos)
    {
        names.push_back(algo->getName());
    }

    //TODO plotPeaknumToTracknum
    plotPeaknumToTracknum(
            names,
            peaknumToTracknum,
            "peaknumToTracknum.pdf");
}

// vim: set ts=4 sw=4 sts=4 expandtab:
