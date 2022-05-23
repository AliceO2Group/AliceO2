#!/usr/bin/env python

from subprocess import call
import os
import argparse
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import numpy as np
import pandas as pd


def gencode_clang_format(filename):
    """ Run clang-format on file """
    clang_format = ["clang-format", "-i", filename]
    return call(clang_format)


def gencode_open_generated(filename):
    """ Open a new file and add a Copyright on it """
    out = open(filename, "w")
    gencode_generated_code(out)
    return out


def gencode_close_generated(out):
    """ Format and close """
    out.close()
    gencode_clang_format(out.name)


def gencode_generated_code(out):
    """ Add full O2 Copyright to out"""
    out.write('''// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

              ///
              /// GENERATED CODE ! DO NOT EDIT !
              ///
              ''')


def gencode_insert_row_in_map(out, row):
    def insert_in_map(dsid, index):
        out.write("add(e2d,{},{},{},{},{});\n"
                  .format(row.de_id, dsid, row.solar_id, row.group_id, index))
    if int(row.ds_id_0):
        insert_in_map(row.ds_id_0, 0)
    if int(row.ds_id_1):
        insert_in_map(row.ds_id_1, 1)
    if int(row.ds_id_2):
        insert_in_map(row.ds_id_2, 2)
    if int(row.ds_id_3):
        insert_in_map(row.ds_id_3, 3)
    if int(row.ds_id_4):
        insert_in_map(row.ds_id_4, 4)


def gencode_do(df, df_cru, solar_map, chamber):
    """ Generate code for one chamber

    Information from the dataframe df is used to create c++ code that
    builds a couple of std::map
    """

    out = gencode_open_generated(chamber + ".cxx")

    out.write('''
              #include "CH.cxx"
              ''')

    out.write(
        "void fillElec2Det{}(std::map<uint32_t,uint32_t>& e2d){{".format(chamber))

    for row in df.itertuples():
        gencode_insert_row_in_map(out, row)

    out.write("}")

    out.write(
        "void fillSolar2FeeLink{}(std::map<uint16_t, uint32_t>& s2f){{".format(chamber))

    for row in df_cru.itertuples():
        if len(row.solar_id) > 0:
            out.write("add_cru(s2f,{},{},{});\n".format(
                row.fee_id, int(row.link_id) % 12, row.solar_id))

    out.write("}")
    gencode_close_generated(out)


def gs_read_sheet(credential_file, workbook, sheet_name):
    """ Read a Google Spreadsheet

    """

    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        credential_file, scope)  # Your json file here

    gc = gspread.authorize(credentials)

    wks = gc.open(workbook).worksheet(sheet_name)

    data = wks.get_all_values()

    cols = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    df = pd.DataFrame(np.asarray(data)[:, cols], columns=["cru", "fiber", "crate", "solar",
                                                          "solar_local_id", "j", "solar_id", "flat_cable_name",
                                                          "length", "de",
                                                          "ds1", "ds2", "ds3", "ds4", "ds5"])
    return df.iloc[3:]


def gs_read_sheet_cru(credential_file, workbook, sheet_name):
    """ Read a Google Spreadsheet

    """

    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        credential_file, scope)  # Your json file here

    gc = gspread.authorize(credentials)

    wks = gc.open(workbook).worksheet(sheet_name)

    data = wks.get_all_values()

# LINK ID  CRU ID  CRU LINK  DWP  CRU ADDR  DW ADDR   FEE ID

    cols = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    df = pd.DataFrame(np.asarray(data)[:, cols],
                      columns=["solar_id", "cru_id", "link_id", "cru_sn",
                               "dwp", "cru_address_0", "cru_address_1",
                               "fee_id"])

    return df.iloc[1:]


def excel_get_dataframe(filename, sheet):
    """ Read a dataframe from an excel file """

    f = pd.read_excel(filename, sheet_name=sheet,
                      names=["cru", "fiber", "crate", "solar",
                             "solar_local_id", "j", "slat",
                             "length", "de",
                             "ds1", "ds2", "ds3", "ds4", "ds5"],
                      usecols="A:N",
                      na_values=[" "],
                      na_filter=True)
    return f


def excel_is_valid_file(excel_parser, arg, sheet):
    print(arg, sheet)
    if not os.path.isfile(arg):
        return excel_parser.error("The file %s does not exist!" % arg)
    return excel_get_dataframe(arg, sheet)


def _simplify_dataframe(df):
    """ Do some cleanup on the dataframe """

    # remove lines where only the "CRATE #" column is
    # different from NaN
    df = df[df.crate != ""]

    # row_list is a dictionary where we'll put only the information we need
    # from the input DataFrame (df)
    row_list = []

    solar_map = {}

    for row in df.itertuples():
        crate = int(str(row.crate).strip('C '))
        solar_pos = int(row.solar.split('-')[2].strip('S '))-1
        group_id = int(row.solar.split('-')[3].strip('J '))-1
        solar_id = crate*8 + solar_pos
        de_id = int(row.de)
        d = dict({
            'cru_id': row.cru,
            'solar_id': solar_id,
            'group_id': group_id,
            'de_id': de_id,
            'ds_id_0': int(row.ds1) if pd.notna(row.ds1) and len(row.ds1) > 0 else 0
        })
        d['ds_id_1'] = int(row.ds2) if pd.notna(
            row.ds2) and len(row.ds2) > 0 else 0
        d['ds_id_2'] = int(row.ds3) if pd.notna(
            row.ds3) and len(row.ds3) > 0 else 0
        d['ds_id_3'] = int(row.ds4) if pd.notna(
            row.ds4) and len(row.ds4) > 0 else 0
        d['ds_id_4'] = int(row.ds5) if pd.notna(
            row.ds5) and len(row.ds5) > 0 else 0
        solar_map[solar_id] = de_id
        row_list.append(d)

    # create the output DataFrame (sf) from the row_list dict
    sf = pd.DataFrame(row_list, dtype=np.dtype('U2'))

    return sf, solar_map


parser = argparse.ArgumentParser()

parser.add_argument('--excel', '-e', dest="excel_filename",
                    action="append",
                    help="input excel filename(s)")

parser.add_argument('--google_sheet', '-gs', dest="gs_name",
                    help="input google sheet name")

parser.add_argument('-s', '--sheet',
                    dest="sheet",
                    required=True,
                    help="name of the excel sheet to consider in the excel file")

parser.add_argument('-c', '--chamber',
                    dest="chamber",
                    help="output c++ code for chamber")

parser.add_argument('--verbose', '-v',
                    dest="verbose", default=False, action="store_true",
                    help="verbose")

parser.add_argument('--credentials',
                    dest="credentials",
                    help="json credential file for Google Sheet API access")

parser.add_argument("--fec_map", "-f",
                    dest="fecmapfile",
                    help="fec.map output filename")

parser.add_argument("--cru_map",
                    dest="crumapfile",
                    help="cru.map output filename")

args = parser.parse_args()

df = pd.DataFrame()
df_cru = pd.DataFrame()

if args.excel_filename:
    for ifile in args.excel_filename:
        df = pd.concat([df, excel_is_valid_file(parser, ifile, args.sheet)])

if args.gs_name:
    df = pd.concat(
        [df, gs_read_sheet(args.credentials, args.gs_name, args.sheet)])
    df, solar_map = _simplify_dataframe(df)
    df_cru = pd.concat([df_cru, gs_read_sheet_cru(args.credentials, args.gs_name,
                                                  args.sheet+" CRU map")])

if args.verbose:
    print(df.to_string())

if args.chamber:
    gencode_do(df, df_cru, solar_map, args.chamber)

if args.fecmapfile:
    fec_string = df.to_string(
        columns=["solar_id", "group_id", "de_id", "ds_id_0",
                 "ds_id_1", "ds_id_2", "ds_id_3", "ds_id_4"],
        header=False,
        index=False,
        formatters={
            "solar_id": lambda x: "%-6s" % x,
            "group_id": lambda x: "%2s" % x,
            "de_id": lambda x: "%9s   " % x,
            "ds_id_0": lambda x: " %-6s" % x,
            "ds_id_1": lambda x: " %-6s" % x,
            "ds_id_2": lambda x: " %-6s" % x,
            "ds_id_3": lambda x: " %-6s" % x,
            "ds_id_4": lambda x: (" %-6s" % x).rstrip(),
        })
    fec_file = open(args.fecmapfile, "w")
    fec_file.write(fec_string.rstrip()+"\n")

if args.crumapfile:
    cru_string = df_cru.to_string(
        columns=["solar_id", "fee_id", "link_id"],
        header=False,
        index=False,
        formatters={
            "solar_id": lambda x: "%4s" % x if x else "XXXX",
            "fee_id": lambda x: "%4s" % x if x else "XXXX",
            "link_id": lambda x: "%4s" % x if x else "XXXX",
        })
    cru_file = open(args.crumapfile, "w")
    [cru_file.write(line.rstrip()+"\n")
     for line in cru_string.split("\n") if not line.startswith("XXXX")]
