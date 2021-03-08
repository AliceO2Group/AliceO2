#!/usr/bin/env python3

# A simple python script producing a JSON
# containing details of O2 software dependencies
# in the alibuild/alidist stack during build.
# The json can be used as a single and fast place
# to query for instance which Geant4, etc., this O2 
# was compiled against.

# Alibuild provides environment variables _REVISION _VERSION
# which tell us the software packages going into the build.

import os
import re

# check if we found a possible Alibuild Entry
# the package name is returned in variable package
def isAlibuildEntry(key):
    m = re.match('(.*)_(VERSION)',key)
    if m!=None:
       package=m.group(1)
       entry="VERSION"
       return (package, entry)
    m = re.match('(.*)_HASH',key)
    if m!=None:
       package=m.group(1)
       entry="HASH"
       return (package, entry)

    return None

# we record information in this dictionary
build_dict={}

for e in os.environ:
    check = isAlibuildEntry(e)
    if check!=None:
       package=check[0]
       entry=check[1]
       packageentry=build_dict.get(package)
       if packageentry != None:
           packageentry[entry]=os.environ[e]
       else:
           build_dict[package]={ entry:os.environ[e] }

# only keep entries that have all wanted features
final_build_dict = { e:v for e,v in build_dict.items() if len(v) == 2 }

import json
with open('o2-build-database.json', 'w') as outfile:
    json.dump(final_build_dict, outfile, indent=2)

