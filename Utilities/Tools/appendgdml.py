#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser(description='Append sensitive volumne information to a GDML file')

# the run-number of data taking or default if unanchored
parser.add_argument('-i', '--input', type=str, help="name of input GDML file")
parser.add_argument('-o', '--output', type=str, default="out.gdml", help="name of input GDML file")
args = parser.parse_args()

# Parse the XML file
tree = ET.parse(args.input)
root = tree.getroot()

# this is where we keep sensitive volume information
file_path="MCStepLoggerSenVol.dat"

sensitiveSet = set()
with open(file_path, 'r') as file:
    for line in file:
        volID, volName = line.strip().split(":")
        # read in the sensitive volumes into a hashmap
        sensitiveSet.add(volName)

# Define the auxiliary element
auxiliary_element = ET.Element('auxiliary', auxtype="SensDet")

# Find all <volume> nodes matching the name of sensitive volume
# and insert the auxiliary element into each one
for volume_node in root.findall('.//volume'):
    if volume_node.get("name") in sensitiveSet:
       volume_node.append(auxiliary_element)

# Write the modified XML back to a file
tree.write(args.output)
