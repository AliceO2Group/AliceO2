#!/usr/bin/env python3

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
# Since: 2026-08

"""Raytrace a TGeo geometry through the real navigator and write a PNG.

One ray per pixel through gGeoManager: InitTrack from the camera plane, step
with FindNextBoundaryAndStep until a daughter of the top volume is entered,
then shade from FindNormal(). The picture is therefore made by exactly the
code the transport uses -- a solid that does not navigate renders as
background, and a TGeoTessellated renders as its bounding box.

Framing is two-pass: a coarse cast finds which pixels hit the subject, and the
real render frames tightly on that. A stray part far from the rest of a model
therefore cannot shrink the subject into a corner.

With --csg-report, each volume is coloured by the representation the cascade
gave it (CSG / exact surfaces / mesh); --grey paints everything uniformly, for
a "before" panel.

    python3 renderTGeo.py geom.root out.png --csg-report csg_report.json
    python3 renderTGeo.py geom.root out.png --grey --theta 66 --phi 28
"""
import argparse, json, math, sys
import numpy as np
import ROOT
from PIL import Image

ROOT.gROOT.SetBatch(True)


def unit(v):
    return v / np.linalg.norm(v)


def render(geofile, out, tiers=None, width=1100, height=800,
           theta=62.0, phi=32.0, bg=(255, 255, 255), pad=1.18,
           grey=False):
    ROOT.TGeoManager.Import(geofile)
    gm = ROOT.gGeoManager
    top = gm.GetTopVolume()

    # --- collect the daughters and their world-frame bounding boxes ---
    names, centres, halfs = [], [], []
    for i in range(top.GetNdaughters()):
        node = top.GetNode(i)
        vol = node.GetVolume()
        box = vol.GetShape()
        tr = node.GetMatrix().GetTranslation()
        names.append(vol.GetName())
        centres.append([tr[0], tr[1], tr[2]])
        halfs.append([box.GetDX(), box.GetDY(), box.GetDZ()])
    centres = np.array(centres)
    halfs = np.array(halfs)
    lo = (centres - halfs).min(axis=0)
    hi = (centres + halfs).max(axis=0)
    centre = 0.5 * (lo + hi)
    radius = 0.5 * np.linalg.norm(hi - lo)

    # --- camera ---
    th, ph = math.radians(theta), math.radians(phi)
    eye_dir = np.array([math.sin(th) * math.cos(ph),
                        math.sin(th) * math.sin(ph),
                        math.cos(th)])
    dist = radius * 3.2
    eye = centre + eye_dir * dist
    fwd = unit(centre - eye)
    up0 = np.array([0.0, 0.0, 1.0])
    right = unit(np.cross(fwd, up0))
    up = unit(np.cross(right, fwd))

    # frame on the projected bbox corners of the daughters within 3x the median distance of the
    # cluster, so a stray part cannot shrink the subject
    corners = []
    for c, h in zip(centres, halfs):
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(c + np.array([sx * h[0], sy * h[1], sz * h[2]]))
    corners = np.array(corners) - eye
    u = corners @ right
    v = corners @ up
    # robust bounds: a single stray part in the CAD model must not shrink the subject
    ulo, uhi = u.min(), u.max()
    vlo, vhi = v.min(), v.max()
    umid, vmid = 0.5 * (ulo + uhi), 0.5 * (vlo + vhi)
    half_u = 0.5 * (uhi - ulo) * pad
    half_v = 0.5 * (vhi - vlo) * pad
    aspect = width / height
    if half_u / half_v < aspect:
        half_u = half_v * aspect
    else:
        half_v = half_u / aspect
    window = (umid - half_u, umid + half_u, vmid - half_v, vmid + half_v)
    light = unit(np.array([0.45, 0.35, 0.82]))

    def cast(win, w, h):
        """Cast one ray per pixel over the camera window; return the image and
        the (u, v) extent of the pixels that actually hit something."""
        u0, u1, v0, v1 = win
        gx = np.linspace(u0, u1, w)
        gy = np.linspace(v1, v0, h)
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[:, :] = bg
        hit_u, hit_v = [], []
        nav = gm.GetCurrentNavigator()
        for iy, sy in enumerate(gy):
            for ix, sx in enumerate(gx):
                o = eye + right * sx + up * sy
                nav.InitTrack(o[0], o[1], o[2], fwd[0], fwd[1], fwd[2])
                nm = None
                for _ in range(24):
                    nav.FindNextBoundaryAndStep()
                    if nav.IsOutside():
                        break
                    cand = nav.GetCurrentVolume().GetName()
                    if cand in vol_colour:
                        nm = cand
                        break
                if nm is None:
                    continue
                hit_u.append(sx); hit_v.append(sy)
                nr = nav.FindNormal()
                n = np.array([nr[0], nr[1], nr[2]])
                nn = np.linalg.norm(n)
                lam = 0.7 if nn == 0 else abs(float(np.dot(n / nn, light)))
                shade = 0.32 + 0.68 * lam
                base = np.array(vol_colour[nm], dtype=float)
                out[iy, ix] = np.clip(base * shade + 45.0 * (shade ** 6), 0, 255)
        return out, (hit_u, hit_v)

    # --- colour per volume ---
    PALETTE = {
        "csg":     (0x2f, 0x6b, 0x4c),
        "surface": (0x1c, 0x62, 0x96),
        "mesh":    (0x9a, 0x5c, 0x17),
    }
    GREY = (0x8a, 0x91, 0x97)
    vol_colour = {}
    for n in names:
        if grey or tiers is None:
            vol_colour[n] = GREY
        else:
            vol_colour[n] = PALETTE.get(tiers.get(n, "mesh"), GREY)

    # pass 1: a coarse cast over the generous window, only to find the subject
    _, (hu, hv) = cast(window, 190, 140)
    if hu:
        mu = 0.06 * max(max(hu) - min(hu), 1e-6)
        mv = 0.06 * max(max(hv) - min(hv), 1e-6)
        u0, u1 = min(hu) - mu, max(hu) + mu
        v0, v1 = min(hv) - mv, max(hv) + mv
        cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        hu2, hv2 = 0.5 * (u1 - u0), 0.5 * (v1 - v0)
        if hu2 / hv2 < aspect:
            hu2 = hv2 * aspect
        else:
            hv2 = hu2 / aspect
        window = (cu - hu2, cu + hu2, cv - hv2, cv + hv2)

    # pass 2: the real render, tightly framed on what pass 1 found
    img, _ = cast(window, width, height)

    Image.fromarray(img).save(out)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("geofile")
    ap.add_argument("out")
    ap.add_argument("--csg-report", default=None)
    ap.add_argument("--grey", action="store_true")
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--theta", type=float, default=62.0)
    ap.add_argument("--phi", type=float, default=32.0)
    a = ap.parse_args()

    tiers = None
    if a.csg_report:
        rep = json.load(open(a.csg_report))
        tiers = {}
        for part in rep.get("parts", []):
            nm = part.get("volume") or part.get("name")
            t = part.get("representation")
            if nm:
                tiers[nm] = t
    render(a.geofile, a.out, tiers=tiers, width=a.width, height=a.height,
           theta=a.theta, phi=a.phi, grey=a.grey)
