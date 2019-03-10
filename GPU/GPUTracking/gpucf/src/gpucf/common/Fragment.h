#pragma once

#include <cstddef>


namespace gpucf
{

    struct Fragment
    {
        /**
         * Index of the first digit that is processed with this fragment.
         */
        size_t start;

        /**
         * Number of digits that are already on the device and have been
         * written to the chargeMap, but have not been looked at for clusters.
         */
        size_t backlog;

        /**
         * Number of digits that have to be transferred to the device,
         * written to the chargeMap and have to run the clusterFinder on.
         */
        size_t items;

        /**
         * Number of digits that have to be transferred to the device and
         * written to the chargeMap but are not looked at for peaks yet.
         * These are necessary as the cluster finder has to look up to two
         * timesteps into the future to compute cluster. So if items and
         * backlog contain digits up to timestep t then the future consists of
         * digits from timesteps t+1 and t+2.
         *
         * Future digits are further processed with the next fragment.
         * The future of the last fragment is always zero.
         */
        size_t future;

        Fragment(size_t);
        Fragment(size_t, size_t, size_t, size_t);
    };

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
