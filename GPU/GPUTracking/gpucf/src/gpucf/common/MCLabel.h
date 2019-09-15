// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/common/RawLabel.h>

#include <functional>
#include <iosfwd>


namespace gpucf
{
    
struct MCLabel
{
    short event; 
    short track;

    MCLabel(const RawLabel &l)
        : event(l.event)
        , track(l.track)
    {
    }

    bool operator==(const MCLabel &other) const
    {
        return other.event == event && other.track == track;
    }

    bool operator<(const MCLabel &other) const
    {
        return event < other.event || (event == other.event && track < other.track);
    }

};

std::ostream &operator<<(std::ostream &o, const MCLabel &l);

} // namespace gpucf


namespace std
{
    template <>
    struct hash<gpucf::MCLabel>
    {
        size_t operator()(const gpucf::MCLabel &l) const
        {
            size_t h = size_t(l.event) << (8*sizeof(short)) | size_t(l.track);
            return std::hash<size_t>()(h);
        }
    };
    
} // namespace std

// vim: set ts=4 sw=4 sts=4 expandtab:
