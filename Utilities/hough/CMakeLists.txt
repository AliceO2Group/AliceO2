# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(MODULE_NAME "hough")

o2_setup(NAME ${MODULE_NAME})

set(Exe_Names runHough)

set(Exe_Source runHough.cxx)

set(BUCKET_NAME hough_bucket)

list(LENGTH Exe_Names _length)
math(EXPR _length ${_length}-1)

foreach(_file RANGE 0 ${_length})
  list(GET Exe_Names ${_file} _name)
  list(GET Exe_Source ${_file} _src)
  o2_generate_executable(EXE_NAME
                         ${_name}
                         SOURCES
                         ${_src}
                         BUCKET_NAME
                         ${BUCKET_NAME})
endforeach(_file RANGE 0 ${_length})
