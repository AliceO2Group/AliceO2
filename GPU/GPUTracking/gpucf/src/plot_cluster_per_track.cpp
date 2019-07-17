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


std::unordered_map<MCLabel, size_t> countPeaksPerTrackById(
        View<size_t> peakIds,
        const LabelContainer &labels)
{
    std::unordered_map<MCLabel, size_t> peaksPerTrack;

    for (const MCLabel &label : labels.allLabels())
    {
        peaksPerTrack[label] = 0;
    }

    for (size_t id : peakIds)
    {
        for (const MCLabel &label : labels[id])
        {
            peaksPerTrack[label]++;
        }
    }

    return peaksPerTrack;
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


template<typename T>
class BucketCounter
{

public:

    BucketCounter(const std::set<T> &keys)
    {
        for (const T &key : keys)
        {
            counts[key] = 0;        
        }
    }

    void add(const T &key)
    {
        counts[key]++;
    }

    const std::unordered_map<T, size_t> &map() const
    {
        return counts;
    }

private:

    std::unordered_map<T, size_t> counts;

};

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
                float bq = chargemap[between];

                bool otherIsPeak = peakmap[other];

                removeMe |= otherIsPeak && (oq > q) && (q - bq >= 2);
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
    SectorData<RawDigit> rawdigits = gpucf::read<RawDigit>(args::get(digitfile));
    SectorMap<std::vector<Digit>> digits = Digit::bySector(rawdigits);

    gpucf::log::Info() << "Reading label file " << args::get(labelfile);
    SectorData<RawLabel> rawlabels = gpucf::read<RawLabel>(args::get(labelfile));
    SectorMap<LabelContainer> labels = LabelContainer::bySector(rawlabels);

    gpucf::log::Info() << "Running cluster finder";

    ClusterFinderConfig cfg;
    cfg.qmaxCutoff = 2; 
    ReferenceClusterFinder cf(cfg);
    SectorMap<ReferenceClusterFinder::Result> cluster = cf.runOnSectors(digits);

    SectorMap<PeakCount> peakcounts;
    SectorMap<PeakCount> peakcountsNS;
    for (size_t sector = 0; sector < TPC_SECTORS; sector++)
    {
        std::vector<size_t> peakIds = getPeakIds(cluster[sector].isPeak);
        std::vector<size_t> noiseSuppressionIds = 
            suppressNoise(digits[sector], cluster[sector].isPeak);

        log::Info() << "Sector " << sector << ": "
            << peakIds.size() - noiseSuppressionIds.size()
            << " peaks removed";

        std::unordered_map<MCLabel, size_t> peaksPerTrack   = 
            countPeaksPerTrackById(peakIds, labels[sector]);
        std::unordered_map<MCLabel, size_t> peaksPerTrackNS = 
            countPeaksPerTrackById(noiseSuppressionIds, labels[sector]);


        peakcounts[sector] = {peaksPerTrack};
        peakcountsNS[sector] = {peaksPerTrackNS};
    }

    PeakCount pc(peakcounts);
    PeakCount pcns(peakcountsNS);


    log::Info() << "Without noise supression:\n" << pc;
    log::Info() << "With noise supression:\n" << pcns;

        /* std::vector<PeakCount> peaksPerTrack = */ 
        /*     peaksPerTrackVaryQmaxCutoff(digits, labels, 100); */

        /* plotPeakCounts(peaksPerTrack); */

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
