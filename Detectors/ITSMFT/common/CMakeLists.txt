# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

#add_subdirectory(base) # Included at top level, due to dependency order
add_subdirectory(simulation)
add_subdirectory(reconstruction)
add_subdirectory(workflow)

o2_data_file(COPY data DESTINATION Detectors/ITSMFT)
