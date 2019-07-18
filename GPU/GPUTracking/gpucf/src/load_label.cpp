#include <gpucf/common/LabelContainer.h>
#include <gpucf/common/log.h>
#include <gpucf/common/serialization.h>

#include <args/args.hxx>

#include <iostream>


using namespace gpucf;


int main(int argc, const char *argv[])
{
    args::ArgumentParser parser("");

    args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
    args::ValueFlag<std::string> infile(parser, "IN", "Label file", {'l', "labels"});
    args::ValueFlag<std::string> digitfile(parser, "IN", "Label file", {'d', "digits"});

    try 
    {
        parser.ParseCLI(argc, argv);
    } 
    catch (const args::Help &) 
    {
        std::cerr << parser;
        return 1;
    }

    gpucf::log::Info() << "Reading label file " << args::get(infile);

    SectorMap<std::vector<RawDigit>> rawdigits = gpucf::read<RawDigit>(args::get(digitfile));
    SectorMap<std::vector<RawLabel>> rawLabels = gpucf::read<RawLabel>(args::get(infile));

    for (size_t i = 0; i < 10; i++)
    {
        log::Debug() << rawLabels[0][i].id;
    }

    gpucf::log::Info() << "Creating label container";

    SectorMap<std::vector<Digit>> digits = Digit::bySector(rawdigits);
    SectorMap<LabelContainer> container = LabelContainer::bySector(rawLabels, digits);

    std::unordered_map<MCLabel, size_t> digitsPerTrack;

    for (const LabelContainer &labels : container)
    {
        for (size_t i = 0; i < labels.size(); i++) 
        {
            for (const MCLabel &l : labels[i])
            {
                auto lookup = digitsPerTrack.find(l);
                if (lookup == digitsPerTrack.end())
                {
                    digitsPerTrack[l] = 1;
                }
                else
                {
                    lookup->second++;
                }
            }
        }
    }

    for (auto it : digitsPerTrack)
    {
        log::Info() << it.first << ": " << it.second;
    }


    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
