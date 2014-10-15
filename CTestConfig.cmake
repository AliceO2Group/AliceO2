 ################################################################################
 #    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    #
 #                                                                              #
 #              This software is distributed under the terms of the             # 
 #         GNU Lesser General Public Licence version 3 (LGPL) version 3,        #  
 #                  copied verbatim in the file "LICENSE"                       #
 ################################################################################
# Dashboard is opened for submissions for a 24 hour period starting at
# the specified NIGHLY_START_TIME. Time is specified in 24 hour format.
set(CTEST_PROJECT_NAME "AliceO2")
set(CTEST_NIGHTLY_START_TIME "00:00:00 CEST")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "cdash.gsi.de")
set(CTEST_DROP_LOCATION "/CDash/submit.php?project=AliceO2")
set(CTEST_DROP_SITE_CDASH TRUE)

set(CTEST_TESTING_TIMEOUT 60)
