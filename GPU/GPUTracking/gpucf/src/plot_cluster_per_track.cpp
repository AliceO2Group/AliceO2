#include <gpucf/algorithms/ReferenceClusterFinder.h>
#include <gpucf/common/LabelContainer.h>
#include <gpucf/common/RawDigit.h>
#include <gpucf/common/serialization.h>

#include <args/args.hxx>

#include <TCanvas.h>
#include <TGraph.h>
#include <TMultiGraph.h>

#include <set>


using namespace gpucf;


struct PeakCount
{
    size_t peakNum0 = 0;
    size_t peakNum1 = 0;
    size_t peakNum10 = 0;
    size_t peakNumX = 0;


    PeakCount() = default;

    PeakCount(const std::unordered_map<MCLabel, size_t> &peaksPerTrack)
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
            peakNum10 += pc.peakNum10;
            peakNumX += pc.peakNumX;
        }
    }

    void update(size_t c)
    {
        peakNum0  += (c == 0);
        peakNum1  += (c == 1);
        peakNum10 += (c > 1 && c < 10);
        peakNumX  += (c >= 10);
    }

};


class LabelBucketCounter
{

public:

    LabelBucketCounter(const std::set<MCLabel> &keys)
    {
        for (const MCLabel &key : keys)
        {
            counts[key] = 0; 
        }
    }

    void add(const LabelBucketCounter &other)
    {
        for (const auto &p : other.map())
        {
            counts[p.first] += p.second;
        }
    }

    void add(View<size_t> peakIds, const LabelContainer &labels)
    {
        for (size_t id : peakIds)
        {
            add(labels[id]);
        }
    }

    void add(View<MCLabel> labels)
    {
        for (const MCLabel &label : labels)
        {
            add(label);
        }
    }

    void add(const MCLabel &key)
    {
        counts[key]++;
    }

    const std::unordered_map<MCLabel, int> &map() const
    {
        return counts;
    }

    std::vector<int> getPeaknumToTracknum() const
    {
        size_t maxPeaksPerTrack = 0;
        for (auto &p : counts)
        {
            maxPeaksPerTrack = std::max(maxPeaksPerTrack, size_t(p.second));
        }

        std::vector<int> peaknumToTracknum(maxPeaksPerTrack+1);
        std::fill(peaknumToTracknum.begin(), peaknumToTracknum.end(), 0);

        for (auto &p : counts)
        {
            peaknumToTracknum[p.second]++;
        }

        return peaknumToTracknum;
    }

private:

    std::unordered_map<MCLabel, int> counts;

};

std::ostream &operator<<(std::ostream &o, const PeakCount &pc)
{
    return o << "Number of tracks with...\n"
             << " ... no peaks    -> " << pc.peakNum0 << "\n"
             << " ... one peak    -> " << pc.peakNum1 << "\n"
             << " ... <  10 peaks -> " << pc.peakNum10 << "\n"
             << " ... >= 10 peaks -> " << pc.peakNumX;
}


int clamp(int x, int l, int r)
{
    return std::max(l, std::min(x, r));
}


void plotPeaknumToTracknum(
        View<int> peaknumToTracknum,
        View<int> peaknumToTracknumNS,
        View<int> peaknumToTracknumRaw)
{

    size_t n = std::max(peaknumToTracknum.size(), peaknumToTracknumNS.size());
    n = std::max(n, peaknumToTracknumRaw.size());
    std::vector<int> x(n);

    for (size_t i = 0; i < x.size(); i++)
    {
        x[i] = i;
    }

    TCanvas *c = new TCanvas("c1", "Cluster per Track", 1200, 800);
    TMultiGraph *mg = new TMultiGraph();

    TGraph *g2 = new TGraph(peaknumToTracknumRaw.size(), x.data(), peaknumToTracknumRaw.data());
    g2->SetLineColor(4);
    g2->SetTitle("No filters");
    mg->Add(g2);

    TGraph *g0 = new TGraph(peaknumToTracknum.size(), x.data(), peaknumToTracknum.data());
    g0->SetLineColor(2);
    g0->SetTitle("With qmaxCutoff");
    mg->Add(g0);

    TGraph *g1 = new TGraph(peaknumToTracknumNS.size(), x.data(), peaknumToTracknumNS.data());
    g1->SetLineColor(3);
    g1->SetTitle("5x5 Noise suppresion");
    mg->Add(g1);

    mg->Draw("AL");

    c->BuildLegend();
    c->SaveAs("peaknumToTracknum.pdf");
}


std::unordered_map<MCLabel, size_t> countPeaksPerTrack(
        const SectorMap<ReferenceClusterFinder::Result> &result,
        const SectorMap<LabelContainer> &labels)
{
    std::unordered_map<MCLabel, size_t> peaksPerTrack;

    for (size_t sector = 0; sector < TPC_SECTORS; sector++)
    {
        const LabelContainer &l = labels[sector];
        View<unsigned char> isPeak = result[sector].isPeak;

        ASSERT(l.size() == isPeak.size());

        for (size_t id = 0; id < isPeak.size(); id++)
        {
            bool peak = isPeak[id];
            View<MCLabel> tracks = l[id];

            for (const MCLabel &label : tracks)
            {
                auto lookup = peaksPerTrack.find(label);
                if (lookup == peaksPerTrack.end())
                {
                    peaksPerTrack[label] = peak;
                }
                else
                {
                    lookup->second += peak;
                }
            }
        }
    }

    return peaksPerTrack;
}


std::vector<PeakCount> peaksPerTrackVaryQmaxCutoff(
        const SectorMap<std::vector<Digit>> &digits,
        const SectorMap<LabelContainer>     &labels,
        size_t cutoffMax)
{
    std::vector<PeakCount> counts;

    for (size_t qmaxCutoff = 0; qmaxCutoff <= cutoffMax; qmaxCutoff++)
    {
        gpucf::log::Info() << "Running cluster finder (cutoff = " << qmaxCutoff << ")";
        ClusterFinderConfig cfg;
        cfg.qmaxCutoff = qmaxCutoff; 
        ReferenceClusterFinder cf(cfg);
        SectorMap<ReferenceClusterFinder::Result> cluster = cf.runOnSectors(digits);

        std::unordered_map<MCLabel, size_t> peaksPerTrack = 
            countPeaksPerTrack(cluster, labels);

        PeakCount count(peaksPerTrack);

        counts.push_back(count);
    }

    return counts;
}


void plotPeakCounts(View<PeakCount> counts)
{
    const size_t n = counts.size();
    std::vector<int> x(n);
    std::vector<int> pc0(n);
    std::vector<int> pc1(n);
    std::vector<int> pc10(n);
    std::vector<int> pcX(n);

    for (size_t i = 0; i < n; i++)
    {
        x[i] = i;

        const PeakCount &pc = counts[i];
        pc0[i] = pc.peakNum0;
        pc1[i] = pc.peakNum1;
        pc10[i] = pc.peakNum10;
        pcX[i] = pc.peakNumX;
    }

    TCanvas *c = new TCanvas("c1", "Cluster per Track", 1000, 400);
    TMultiGraph *mg = new TMultiGraph();

    TGraph *g0 = new TGraph(n, x.data(), pc0.data());
    g0->SetLineColor(1);
    /* g0->Draw(); */
    mg->Add(g0);

    TGraph *g1 = new TGraph(n, x.data(), pc1.data());
    g1->SetLineColor(2);
    /* g1->Draw(); */
    mg->Add(g1);

    TGraph *g10 = new TGraph(n, x.data(), pc10.data());
    g10->SetLineColor(3);
    mg->Add(g10);

    TGraph *gX = new TGraph(n, x.data(), pcX.data());
    gX->SetLineColor(4);
    mg->Add(gX);

    mg->Draw("AC");

    c->SaveAs("ClusterPerTrack.png");
}



std::set<MCLabel> makeLabelSet(const SectorMap<LabelContainer> &labels)
{
    std::set<MCLabel> uniqueLabels;

    for (const LabelContainer &container : labels)
    {
        View<MCLabel> ls = container.allLabels();
        uniqueLabels.insert(ls.begin(), ls.end());
    }

    return uniqueLabels;
}


std::vector<size_t> suppressNoise(View<Digit> digits, View<unsigned char> isPeak)
{
    Map<float> chargemap(digits, [](const Digit &d) { return d.charge; }, 0.f);
    Map<unsigned char> peakmap(digits, isPeak, 0);

    std::vector<size_t> notRemoved;

    for (size_t i = 0; i < digits.size(); i++)
    {
        if (!isPeak[i])
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

                Position other(digits[i], dp, dt);
                Position between(digits[i], clamp(dp, -1, 1), clamp(dt, -1, 1));

                float q = digits[i].charge;
                float oq = chargemap[other];
                /* float bq = chargemap[between]; */

                bool otherIsPeak = peakmap[other];

                removeMe |= otherIsPeak && (oq > q); //&& (q - bq <= 2);
            }
        }

        if (!removeMe)
        {
            notRemoved.push_back(i);
        }
    }

    return notRemoved;
}

std::vector<size_t> getPeakIds(View<unsigned char> isPeak)
{
    std::vector<size_t> peakIds;
    peakIds.reserve(isPeak.size());
    for (size_t i = 0; i < isPeak.size(); i++)
    {
        if (isPeak[i])
        {
            peakIds.push_back(i);
        }
    }

    return peakIds;
}


int main(int argc, const char *argv[])
{
    args::ArgumentParser parser("");

    args::ValueFlag<std::string> digitfile(parser, "D", "Digit file", {'d', "digits"});
    args::ValueFlag<std::string> labelfile(parser, "L", "Label file", {'l', "labels"});

    parser.ParseCLI(argc, argv);

    gpucf::log::Info() << "Reading digit file " << args::get(digitfile);
    SectorMap<std::vector<RawDigit>> rawdigits = gpucf::read<RawDigit>(args::get(digitfile));
    SectorMap<std::vector<Digit>> digits = Digit::bySector(rawdigits);

    gpucf::log::Info() << "Reading label file " << args::get(labelfile);
    SectorMap<std::vector<RawLabel>> rawlabels = gpucf::read<RawLabel>(args::get(labelfile));
    SectorMap<LabelContainer> labels = LabelContainer::bySector(rawlabels, digits);

    gpucf::log::Info() << "Running cluster finder";

    ClusterFinderConfig cfg;
    cfg.qmaxCutoff = 2; 
    ReferenceClusterFinder cf(cfg);
    SectorMap<ReferenceClusterFinder::Result> cluster = cf.runOnSectors(digits);

    ClusterFinderConfig cfgRaw;
    cfgRaw.qmaxCutoff = 0; 
    ReferenceClusterFinder cfraw(cfgRaw);
    SectorMap<ReferenceClusterFinder::Result> clusterRaw = cfraw.runOnSectors(digits);

    std::set<MCLabel> uniqueLabels = makeLabelSet(labels);

    LabelBucketCounter peakcounts(uniqueLabels);
    LabelBucketCounter peakcountsRaw(uniqueLabels);
    LabelBucketCounter peakcountsNS(uniqueLabels);
    for (size_t sector = 0; sector < TPC_SECTORS; sector++)
    {
        std::vector<size_t> peakIds = getPeakIds(cluster[sector].isPeak);
        std::vector<size_t> noiseSuppressionIds = 
            suppressNoise(digits[sector], cluster[sector].isPeak);
        std::vector<size_t> rawIds = getPeakIds(clusterRaw[sector].isPeak);

        log::Info() << "Sector " << sector << ": "
            << peakIds.size() - noiseSuppressionIds.size()
            << " peaks removed";

        peakcounts.add(peakIds, labels[sector]);
        peakcountsRaw.add(rawIds, labels[sector]);
        peakcountsNS.add(noiseSuppressionIds, labels[sector]);
    }

    std::vector<int> peaknumToTracknum = peakcounts.getPeaknumToTracknum();
    std::vector<int> peaknumToTracknumRaw = peakcountsRaw.getPeaknumToTracknum();
    std::vector<int> peaknumToTracknumNS = peakcountsNS.getPeaknumToTracknum();

    plotPeaknumToTracknum(peaknumToTracknum, peaknumToTracknumNS, peaknumToTracknumRaw);

    /* for (size_t i = 0; i < peaknumToTracknum.size(); i++) */
    /* { */
    /*     log::Info() << i << ": " << peaknumToTracknum[i]; */
    /* } */

    /* log::Info() << "Without noise supression:\n" << pc; */
    /* log::Info() << "With noise supression:\n" << pcns; */

        /* std::vector<PeakCount> peaksPerTrack = */ 
        /*     peaksPerTrackVaryQmaxCutoff(digits, labels, 100); */

        /* plotPeakCounts(peaksPerTrack); */

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
