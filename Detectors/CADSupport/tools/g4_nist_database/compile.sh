# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.
# Author: Sandro Wenzel <sandro.wenzel@cern.ch>
# Since: 2026-03

echo "Compiling using geant4-config..."

g++ -std=c++20 nist_export_all.cxx \
    $(geant4-config --cflags) \
    $(geant4-config --libs) \
    -O2 -o nist_export_all

echo ""
echo "Build complete."
echo "Run with:"
echo "  ./nist_export_all nist_db_all.json"