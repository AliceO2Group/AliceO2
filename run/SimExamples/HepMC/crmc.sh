#!/bin/sh
# This script _may not_ write to standard out, except for the HepMC
# event record.

crmcParam=$(dirname $(dirname `which crmc`))/etc/crmc.param
exec crmc -c $crmcParam $@ -o hepmc -f /dev/stdout | \
    sed -n 's/^\(HepMC::\|[EAUWVP] \)/\1/p'
