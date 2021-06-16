# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.


install(PROGRAMS sim_performance_test.sh sim_challenge.sh check_embedding_pileup.sh produce_QEDhits.sh
                 full_system_test.sh full_system_test_ci_extra_tests.sh
        DESTINATION prodtests)
install(DIRECTORY full-system-test
        DESTINATION prodtests
        PATTERN *
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)
