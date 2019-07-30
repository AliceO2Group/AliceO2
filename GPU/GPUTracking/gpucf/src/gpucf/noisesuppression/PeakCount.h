#pragma once


namespace gpucf
{

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
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
