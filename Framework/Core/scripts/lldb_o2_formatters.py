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
#
# lldb data formatters for o2::framework types.
#
# Usage: add to ~/.lldbinit or a project .lldbinit:
#   command script import /path/to/O2/Framework/Core/scripts/lldb_o2_formatters.py

import lldb

# o2::framework::VariantType enum values (must match Variant.h)
_VARIANT_TYPE = {
    0:  'Int',
    1:  'Int64',
    2:  'Float',
    3:  'Double',
    4:  'String',
    5:  'Bool',
    6:  'ArrayInt',
    7:  'ArrayFloat',
    8:  'ArrayDouble',
    9:  'ArrayBool',
    10: 'ArrayString',
    11: 'Array2DInt',
    12: 'Array2DFloat',
    13: 'Array2DDouble',
    14: 'LabeledArrayInt',
    15: 'LabeledArrayFloat',
    16: 'LabeledArrayDouble',
    17: 'UInt8',
    18: 'UInt16',
    19: 'UInt32',
    20: 'UInt64',
    21: 'Int8',
    22: 'Int16',
    23: 'LabeledArrayString',
    24: 'Empty',
    25: 'Dict',
    26: 'Unknown',
}

# Map VariantType value → (C type name for FindFirstType, is_pointer_to_value)
# is_pointer_to_value=True means mStore holds a T* pointing to heap data (arrays)
_SIMPLE_TYPES = {
    0:  ('int',           False),
    1:  ('long long',     False),
    2:  ('float',         False),
    3:  ('double',        False),
    5:  ('bool',          False),
    17: ('unsigned char', False),
    18: ('unsigned short',False),
    19: ('unsigned int',  False),
    20: ('unsigned long long', False),
    21: ('signed char',   False),
    22: ('short',         False),
}

_ARRAY_ELEM_TYPES = {
    6:  'int',
    7:  'float',
    8:  'double',
    9:  'bool',
}

MAX_ARRAY_DISPLAY = 16


import struct as _struct


def _read_pointer(process, addr):
    err = lldb.SBError()
    ptr_size = process.GetAddressByteSize()
    data = process.ReadMemory(addr, ptr_size, err)
    if err.Fail() or not data:
        return None
    fmt = '<Q' if ptr_size == 8 else '<I'
    return _struct.unpack(fmt, data)[0]


def _read_libcxx_string(process, addr):
    """Read a libc++ std::string directly from memory (no type lookup needed).

    libc++ std::string on 64-bit little-endian (macOS x86_64 / arm64):
      sizeof = 24 bytes, union of:
        __short: byte[0] = (size<<1)|0  (short flag),  bytes[1..22] = inline data
        __long:  byte[0] low bit = 1,  bytes[8..15] = size,  bytes[16..23] = data ptr
    """
    err = lldb.SBError()
    raw = process.ReadMemory(addr, 24, err)
    if err.Fail() or not raw:
        return '"<read error>"'
    b = raw if isinstance(raw[0], int) else bytes(ord(c) for c in raw)
    if (b[0] & 1) == 0:          # short form
        size = b[0] >> 1
        text = b[1:1 + size].decode('utf-8', errors='replace')
    else:                          # long form
        size     = _struct.unpack_from('<Q', b, 8)[0]
        data_ptr = _struct.unpack_from('<Q', b, 16)[0]
        if data_ptr == 0:
            return '"<null>"'
        heap = process.ReadMemory(data_ptr, min(size, 512), err)
        if err.Fail() or not heap:
            return '"<read error>"'
        h = heap if isinstance(heap[0], int) else bytes(ord(c) for c in heap)
        text = h[:size].decode('utf-8', errors='replace')
        if size > 512:
            text += '...'
    return f'"{text}"'


def variant_summary(valobj, _internal_dict):
    # Use GetNonSyntheticValue() so we see the real struct members even when
    # the synthetic provider has replaced the children with decoded values.
    raw = valobj.GetNonSyntheticValue()
    mType_val = raw.GetChildMemberWithName('mType')
    if not mType_val.IsValid():
        return '<invalid Variant>'

    mType = mType_val.GetValueAsUnsigned(26)  # default Unknown
    mSize = raw.GetChildMemberWithName('mSize').GetValueAsUnsigned(1)
    mStore = raw.GetChildMemberWithName('mStore')
    store_addr = mStore.GetLoadAddress()

    type_name = _VARIANT_TYPE.get(mType, f'<type={mType}>')
    target = valobj.GetTarget()
    process = valobj.GetProcess()

    # --- simple scalar types ---
    if mType in _SIMPLE_TYPES:
        ctype, _ = _SIMPLE_TYPES[mType]
        t = target.FindFirstType(ctype)
        if t.IsValid():
            v = valobj.CreateValueFromAddress('v', store_addr, t)
            return f'{type_name}({v.GetValue()})'
        return f'{type_name}(?)'

    # --- String (const char* stored in mStore) ---
    if mType == 4:
        ptr = _read_pointer(process, store_addr)
        if ptr and ptr != 0:
            s = _read_cstring(process, ptr)
            return f'String("{s}")'
        return 'String(null)'

    # --- C-style numeric arrays (int*, float*, double*, bool*) ---
    if mType in _ARRAY_ELEM_TYPES:
        elem_type_name = _ARRAY_ELEM_TYPES[mType]
        ptr = _read_pointer(process, store_addr)
        if not ptr or ptr == 0:
            return f'{type_name}(null)'
        elem_t = target.FindFirstType(elem_type_name)
        if not elem_t.IsValid():
            return f'{type_name}(? x {mSize})'
        count = min(mSize, MAX_ARRAY_DISPLAY)
        items = []
        for i in range(count):
            v = valobj.CreateValueFromAddress(f'e{i}', ptr + i * elem_t.GetByteSize(), elem_t)
            items.append(v.GetValue() or '?')
        result = f'{type_name}([{", ".join(items)}]'
        if mSize > MAX_ARRAY_DISPLAY:
            result += f', ... ({mSize} total)'
        result += ')'
        return result

    # --- ArrayString: std::vector<std::string> stored via placement new in mStore ---
    if mType == 10:
        # libc++ std::vector layout: __begin_, __end_, __end_cap_ (all pointers)
        # libc++ std::string is always 24 bytes on 64-bit (SSO layout)
        STR_SIZE = 24
        ptr_size = process.GetAddressByteSize()
        begin_ptr = _read_pointer(process, store_addr)
        end_ptr   = _read_pointer(process, store_addr + ptr_size)
        if begin_ptr is None or end_ptr is None:
            return 'ArrayString(?)'

        count = (end_ptr - begin_ptr) // STR_SIZE if end_ptr >= begin_ptr else 0
        items = []
        for i in range(min(count, MAX_ARRAY_DISPLAY)):
            items.append(_read_libcxx_string(process, begin_ptr + i * STR_SIZE))
        result = f'ArrayString([{", ".join(items)}]'
        if count > MAX_ARRAY_DISPLAY:
            result += f', ... ({count} total)'
        result += ')'
        return result

    return f'{type_name}(mSize={mSize})'


class VariantSyntheticProvider:
    """Synthetic children for o2::framework::Variant — exposes decoded value as child."""

    def __init__(self, valobj, _internal_dict):
        self.valobj = valobj
        self.children = []

    def num_children(self):
        return len(self.children)

    def get_child_index(self, name):
        for i, (n, _) in enumerate(self.children):
            if n == name:
                return i
        return -1

    def get_child_at_index(self, index):
        if 0 <= index < len(self.children):
            return self.children[index][1]
        return None

    def update(self):
        self.children = []
        # Use GetNonSyntheticValue() to read the real struct members.
        raw = self.valobj.GetNonSyntheticValue()
        mType = raw.GetChildMemberWithName('mType').GetValueAsUnsigned(26)
        mSize = raw.GetChildMemberWithName('mSize').GetValueAsUnsigned(1)
        mStore = raw.GetChildMemberWithName('mStore')
        store_addr = mStore.GetLoadAddress()
        target = self.valobj.GetTarget()
        process = self.valobj.GetProcess()

        if mType in _SIMPLE_TYPES:
            ctype, _ = _SIMPLE_TYPES[mType]
            t = target.FindFirstType(ctype)
            if t.IsValid():
                v = self.valobj.CreateValueFromAddress('value', store_addr, t)
                self.children.append(('value', v))

        elif mType == 4:  # String
            ptr = _read_pointer(process, store_addr)
            if ptr and ptr != 0:
                char_t = target.FindFirstType('char').GetPointerType()
                v = self.valobj.CreateValueFromAddress('value', store_addr, char_t)
                self.children.append(('value', v))

        elif mType in _ARRAY_ELEM_TYPES:
            elem_type_name = _ARRAY_ELEM_TYPES[mType]
            ptr = _read_pointer(process, store_addr)
            if ptr and ptr != 0:
                elem_t = target.FindFirstType(elem_type_name)
                if elem_t.IsValid():
                    for i in range(min(mSize, MAX_ARRAY_DISPLAY)):
                        v = self.valobj.CreateValueFromAddress(f'[{i}]', ptr + i * elem_t.GetByteSize(), elem_t)
                        self.children.append((f'[{i}]', v))

        elif mType == 10:  # ArrayString
            # std::vector<std::string> via placement new; std::string = 24 bytes (libc++ 64-bit)
            STR_SIZE = 24
            ptr_size = process.GetAddressByteSize()
            begin_ptr = _read_pointer(process, store_addr)
            end_ptr   = _read_pointer(process, store_addr + ptr_size)
            char_t = target.FindFirstType('char')
            if begin_ptr is not None and end_ptr is not None and end_ptr >= begin_ptr and char_t.IsValid():
                count = (end_ptr - begin_ptr) // STR_SIZE
                err = lldb.SBError()
                for i in range(min(count, MAX_ARRAY_DISPLAY)):
                    str_addr = begin_ptr + i * STR_SIZE
                    raw = process.ReadMemory(str_addr, STR_SIZE, err)
                    if err.Fail() or not raw:
                        continue
                    b = raw if isinstance(raw[0], int) else bytes(ord(c) for c in raw)
                    if (b[0] & 1) == 0:  # short form: data inline at offset 1
                        data_addr = str_addr + 1
                        sz = max(b[0] >> 1, 1)
                    else:               # long form: data pointer at offset 16
                        data_addr = _struct.unpack_from('<Q', b, 16)[0]
                        sz = max(_struct.unpack_from('<Q', b, 8)[0], 1)
                    # Expose as char[sz] so lldb renders it as a string literal
                    arr_t = char_t.GetArrayType(min(sz + 1, 256))
                    sv = self.valobj.CreateValueFromAddress(f'[{i}]', data_addr, arr_t)
                    self.children.append((f'[{i}]', sv))

        return False  # no pruning needed

    def has_children(self):
        return len(self.children) > 0


def __lldb_init_module(debugger, _internal_dict):
    debugger.HandleCommand(
        'type summary add -x "^o2::framework::Variant$" '
        '--python-function lldb_o2_formatters.variant_summary'
    )
    debugger.HandleCommand(
        'type synthetic add -x "^o2::framework::Variant$" '
        '--python-class lldb_o2_formatters.VariantSyntheticProvider'
    )
    print('o2::framework::Variant formatters loaded.')
