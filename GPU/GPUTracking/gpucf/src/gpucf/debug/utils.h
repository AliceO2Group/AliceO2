#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/common/Map.h>
#include <gpucf/common/View.h>

#include <sstream>
#include <vector>


namespace gpucf
{

    template<typename T>
    using Array2D = std::vector<std::vector<T>>;

    template<typename T>
    size_t getWidthPad(const Array2D<T> &data)
    {
        return data.size();
    }

    template<typename T>
    size_t getWidthTime(const Array2D<T> &data)
    {
        size_t w = 0;
        for (const auto &v : data)
        {
            w = std::max(w, v.size());
        }
        return w;
    }

    std::vector<Digit> digitize(const Array2D<float> &);

    template<typename T>
    Map<T> mapify(
            View<T> data,
            T fallback,
            size_t widthPads,
            size_t widthTime)
    {
        Map<T> map(fallback);

        for (size_t time = 0; time < widthTime; time++)
        {
            for (size_t pad = 0; pad < widthPads; pad++)
            {
                size_t idx = TPC_NUM_OF_PADS * (time + PADDING) 
                    + tpcGlobalPadIdx(0, pad);

                if (data[idx] != fallback)
                {
                    map.insert({0, pad_t(pad), timestamp(time)}, data[idx]);
                }
            }
        }

        return map;
    }

    template<typename T>
    Map<T> mapify(const Array2D<T> &data, T fallback)
    {
        Map<T> map(fallback);

        for (size_t pad = 0; pad < data.size(); pad++)
        {
            for (size_t time = 0; time < data[pad].size(); time++)
            {
                if (data[pad][time] != fallback)
                {
                    map.insert({0, pad, time}, data[pad][time]);
                }
            }
        }

        return map;
    }

    template<typename T>
    std::string print(const Map<T> &map, size_t pads, size_t timebins)
    {
        std::stringstream ss;
        for (size_t t = 0; t < timebins; t++)
        {
            for (size_t p = 0; p < pads; p++)
            {
                ss << int(map[{0, pad_t(p), timestamp(t)}]) << " ";
            }
            ss << "\n";
        }

        return ss.str();
    }

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
