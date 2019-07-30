#include <gpucf/algorithms/cpu.h>
#include <gpucf/common/LabelContainer.h>
#include <gpucf/common/log.h>
#include <gpucf/common/RowMap.h>
#include <gpucf/common/SectorMap.h>
#include <gpucf/common/serialization.h>
#include <gpucf/common/TpcHitPos.h>
#include <gpucf/noisesuppression/NoiseSuppressionOverArea.h>

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
    size_t noPeaks = 0;
    size_t onePeak = 0;
    size_t twoPeaksOverlap = 0;
    size_t twoPeaksNoOverlap = 0;
    size_t tenPeaks = 0;
    size_t moreThanTenPeaks = 0;
};

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
    for (size_t i = 0; i < 100; i++)
    {
        algorithms.emplace_back(2, 3, 4, i);
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

    log::Info() << "Creating chargemap";
    for (size_t sector = 0; sector < TPC_SECTORS; sector++)
    {
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
                alg->run(peaks[sector], peakmap, chargemaps[sector]);
        }

    }
    
    // TODO sort hits

    // TODO plot epsilon -> hits
}

// vim: set ts=4 sw=4 sts=4 expandtab:
