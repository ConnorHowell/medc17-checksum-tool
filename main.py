#!/usr/bin/env python3
"""
MEDC17 Checksum Tool v1.1

Professional checksum analyzer and corrector for MEDC17 ECU binaries.
Supports CRC32, ADD32, and ADD16 algorithms with mathematical GF(2) solving.
Includes CVN (Calibration Verification Number) calculation and correction.

I am aware this is an unholy amount of lines of code, I will probably separate this out in the future; or not (probably not)
This could probably be massively simplified, and will likely draw critique; but last I checked there's no other open source checksum correction tools for these ECUs 🤷‍♂️

Copyright (c) 2025 Connor Howell
Licensed under the MIT License
"""

import json
import os
import re
import struct
import sys
import bisect
import hashlib
from types import SimpleNamespace
from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path

# rich is optional: the pretty CLI report uses it, --json never does. The plain
# stand-ins below keep every mode working with nothing but the standard library.
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.text import Text
except ImportError:
    _MARKUP_RE = re.compile(r'\[/?[a-z0-9 #_=\.\-]{0,40}\]', re.IGNORECASE)

    def _strip_markup(value) -> str:
        return _MARKUP_RE.sub('', str(value))

    class Console:
        def print(self, *values, **_kwargs):
            print(' '.join(_strip_markup(v) for v in values) if values else '')

    class Panel:
        def __init__(self, content, title=None, **_kwargs):
            self.content = content
            self.title = title

        def __str__(self):
            heading = f"--- {_strip_markup(self.title)} ---\n" if self.title else ''
            return heading + _strip_markup(self.content)

    class Table:
        def __init__(self, title=None, **_kwargs):
            self.title = title
            self.rows = []

        def add_column(self, *_args, **_kwargs):
            pass

        def add_row(self, *cells):
            self.rows.append([_strip_markup(c).replace('\n', ' ') for c in cells])

        def __str__(self):
            lines = [f"--- {_strip_markup(self.title)} ---"] if self.title else []
            lines.extend('  '.join(row) for row in self.rows)
            return '\n'.join(lines)

    def Text(value, style=None):
        return _strip_markup(value)

    box = SimpleNamespace(SIMPLE=None, ROUNDED=None)


class _NullConsole:
    """Swallows every human-readable message, so --json owns stdout."""

    def print(self, *_values, **_kwargs):
        pass


console = Console()

# --json emits one JSON document on stdout and nothing else
QUIET = False


def set_quiet(quiet: bool) -> None:
    global QUIET, console
    QUIET = quiet
    console = _NullConsole() if quiet else Console()


def print_banner():
    """Display tool banner with version info."""
    banner_text = """
================================================================
          MEDC17 Checksum Analyzer & Corrector v1.1
================================================================
    """
    console.print(banner_text, style="bold cyan")
    console.print("Advanced checksum tool for Bosch MED/EDC17 ECU binaries", style="dim")
    console.print("Supports: CRC32 (GF(2) solver), ADD32, ADD16, CVN\n", style="dim")


def print_success(message: str):
    console.print(f"✓ {message}", style="bold green")


def print_error(message: str):
    console.print(f"✗ {message}", style="bold red")


def print_info(message: str):
    console.print(f"ℹ {message}", style="blue")


def print_warning(message: str):
    console.print(f"⚠ {message}", style="yellow")


# CRC32 lookup table, IEEE 802.3 reflected poly 0xEDB88320
CRC32_TABLE = None

def init_crc32_table():
    global CRC32_TABLE
    if CRC32_TABLE is not None:
        return

    CRC32_TABLE = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
        CRC32_TABLE.append(crc)


# Reverse CRC32 table. Indexed by the top byte of the running register, it walks a
# CRC32 backwards from the end of a region: rev[fwd[j] >> 24] = (fwd[j] << 8) ^ j.
CRC32_REVERSE_TABLE = None

def init_crc32_reverse_table():
    global CRC32_REVERSE_TABLE
    if CRC32_REVERSE_TABLE is not None:
        return
    init_crc32_table()
    rev = [0] * 256
    for j in range(256):
        fwd = CRC32_TABLE[j]
        rev[fwd >> 24] = ((fwd << 8) & 0xFFFFFFFF) ^ j
    CRC32_REVERSE_TABLE = rev


def crc32_fold_patch(data, region_start: int, region_end_incl: int, hole: int,
                     init: int = 0xFFFFFFFF, target: int = 0,
                     xorout: int = 0xFFFFFFFF) -> int:
    """Solve the 4 bytes at `hole` so CRC32 over [region_start, region_end_incl]
    equals `target`. Returned as an int, to store little-endian at `hole`.

    Reflected CRC32 with the given init/xorout; init=xorout=0xFFFFFFFF is plain
    zlib.crc32. Runs the CRC forward from the region start and backward from the
    region end, then folds the two registers into the hole dword.
    """
    init_crc32_reverse_table()
    fwd_tbl = CRC32_TABLE
    rev_tbl = CRC32_REVERSE_TABLE

    # Forward to the near edge of the hole
    f = init & 0xFFFFFFFF
    for pos in range(region_start, hole):
        f = (f >> 8) ^ fwd_tbl[(data[pos] ^ f) & 0xFF]

    # Backward from the region end to the far edge of the hole
    u = (target ^ xorout) & 0xFFFFFFFF
    pos = region_end_incl
    while hole + 4 <= pos:
        u = (data[pos] ^ rev_tbl[u >> 24] ^ ((u << 8) & 0xFFFFFFFF)) & 0xFFFFFFFF
        pos -= 1

    # Fold forward into backward, byte by byte
    u = ((u << 8) & 0xFFFFFFFF) ^ rev_tbl[u >> 24]
    u = (((f >> 16) & 0xFF) ^ rev_tbl[u >> 24] ^ (((u ^ (f >> 24)) << 8) & 0xFFFFFFFF)) & 0xFFFFFFFF
    u = (((f >> 8) & 0xFF) ^ rev_tbl[u >> 24] ^ ((u << 8) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return (((u << 8) & 0xFFFFFFFF) ^ rev_tbl[u >> 24] ^ (f & 0xFF)) & 0xFFFFFFFF


# Erased/free dword fills: NOR-erased 0xFF, zeroed, or the Bosch 0xC3 pad. One of
# these can be overwritten with a compensation value without clobbering live data.
ERASED_DWORDS = (0x00000000, 0xFFFFFFFF, 0xC3C3C3C3)


# Block identifiers observed in binaries
BLOCK_IDENTIFIERS = {
    0x10: 'Startup Block',
    0x20: 'Tuning protection',
    0x30: 'Customer Block',
    0x40: 'Application software #0',
    0x50: 'Application software #1',
    0x60: 'Dataset #0',
    0x70: 'Dataset #1',
    0x80: 'Variant dataset',
    0x90: 'Customer Tuning protection',
    0xA0: 'Application software #2',
    0xB0: 'Application software #3',
    0xC0: 'Absolute constants #0',
    0xD0: 'Emulation extension chip',
    0xE0: 'Customer specific',
    0xF0: 'Ramloader',
    0xF1: 'Application Attestation',
}


# MED17/EDC17 block signature ------------------------------------------------
# Every block seeds its first 32-byte checksum structure with the magic pair
# FADECAFE / CAFEAFFE, at a fixed distance from the block start:
#   block_start + 0x34 = first checksum structure
#                +0x0C = seed      -> 0xFADECAFE at block_start + 64
#                +0x10 = expected  -> 0xCAFEAFFE at block_start + 68
# A 1-in-2^64 anchor, so blocks are found by searching for the pair directly.
MEDC17_MAGIC_A = 0xFADECAFE           # checksum seed   (block_start + 64)
MEDC17_MAGIC_B = 0xCAFEAFFE           # expected value  (block_start + 68)
MEDC17_MAGIC_PAIR = struct.pack('<II', MEDC17_MAGIC_A, MEDC17_MAGIC_B)
MEDC17_MAGIC_OFFSET = 64              # distance from block start to magic word A
# Block class byte (block_start + 15, the high byte of the block-end address):
#   0x80 = application (cached flash view)  0xA0 = data/calibration (uncached)
MEDC17_BLOCK_CLASSES = (0x80, 0xA0)

# "Not programmed" fill words. A reserved partition leaves checksum/adjust as a
# repeating fill byte (0xFF on plain NOR flash, 0xAF in some Bosch images) instead
# of computed values — nothing real to protect, so correction must skip it.
MEDC17_BLANK_WORDS = (0xFFFFFFFF, 0xAFAFAFAF)

# TriCore aliases the same physical flash into cached 0x8xxxxxxx and uncached
# 0xAxxxxxxx views, and a block may use either. Strip the segment nibble before
# converting to a file offset or a 0xA0 (data/cal) block lands 0x20000000 out.
FLASH_SEGMENT_MASK = 0x0FFFFFFF          # offset within the 256 MB flash segment


# Variant dataset ------------------------------------------------------------
# ECUs with more than one calibration variant keep a table of parameter
# addresses per variant. The monitoring checksum is calculated over whichever
# variant is active, so every variant needs its own compensation value —
# correcting only the first leaves the rest wrong.
VARIANT_BLOCK_ID = 0x80
VARIANT_COUNT_OFFSET = 0x7C           # variant count (main + overrides) in the 0x80 block header
VARIANT_MONITOR_MARKER = b'\x61\x04\x04\x00'   # first 4 bytes of the monitored structure
VARIANT_TABLE_MIN_RUN = 51            # shortest ascending run accepted as an address table
VARIANT_TABLE_MIN_SPAN = 0x10000      # smallest address span an address table may cover
VARIANT_TABLE_MIN_MATCH = 76          # percent of dwords a copy must share with the base table
VARIANT_MAX_TABLES = 9
VARIANT_REMAP_SENTINEL = 0x00FFFFFF   # entry value that aborts a table walk


def flash_canonical(mem_addr: int) -> int:
    """Cached-segment form of a flash address, so the 0x80 and 0xA0 aliases of the
    same physical byte compare and translate identically."""
    return (mem_addr & FLASH_SEGMENT_MASK) | 0x80000000


def hex32(value: Optional[int]) -> Optional[str]:
    """0x-prefixed 32-bit hex, or None so JSON consumers see a null."""
    if value is None:
        return None
    return f"0x{value & 0xFFFFFFFF:08X}"


def cube_root_int(n):
    """Integer cube root, Newton's method."""
    if n == 0:
        return 0
    x = n
    y = (2 * x + n // (x * x)) // 3
    while y < x:
        x = y
        y = (2 * x + n // (x * x)) // 3
    return x


def forge_bleichenbacher_signature(ripemd_hash: bytes) -> bytes:
    """Forge a Bleichenbacher e=3 signature carrying a 20-byte RIPEMD-160 hash.

    Simplified format, not full PKCS#1 v1.5: no modulus (verification just cubes
    the signature), and the cube reads 01 FF*8 00 [hash] [garbage] — leading 01
    rather than 00 01, raw hash with no DigestInfo wrapper. Returns 128 bytes.
    """
    # 127 bytes, not 128, to avoid a leading 00 in the cube
    target = bytearray(127)
    target[0] = 0x01
    for i in range(1, 9):
        target[i] = 0xFF
    target[9] = 0x00
    target[10:30] = ripemd_hash
    # Rest is don't-care

    target_int = int.from_bytes(bytes(target), 'big')
    sig_int = cube_root_int(target_int)

    # Largest candidate whose cube still fits under the target
    best_sig = sig_int
    for candidate in [sig_int - 1, sig_int, sig_int + 1, sig_int + 2]:
        if candidate < 0:
            continue
        cubed = candidate ** 3
        if cubed <= target_int and candidate > best_sig:
            best_sig = candidate

    sig_bytes = best_sig.to_bytes(128, 'big')
    return sig_bytes


def crc32_process_dword_bitwise(initial_crc: int, dword_input: int) -> int:
    """Bit-by-bit CRC32 over a single dword."""
    crc = initial_crc
    dword = dword_input
    for _ in range(32):
        xor_result = dword ^ crc
        dword >>= 1
        if (xor_result & 1) != 0:
            crc = (crc >> 1) ^ 0xEDB88320
        else:
            crc = crc >> 1
    return crc


def build_crc_transformation_matrix(intermediate_crc: int) -> list:
    """32x32 GF(2) matrix: effect of each input dword bit on the output CRC.

    Returns 32 rows as bitmasks.
    """
    baseline = crc32_process_dword_bitwise(intermediate_crc, 0)
    matrix = []

    for output_bit in range(32):
        row_value = 0
        for input_bit in range(32):
            test_dword = 1 << input_bit
            result = crc32_process_dword_bitwise(intermediate_crc, test_dword)
            effect = result ^ baseline
            if (effect >> output_bit) & 1:
                row_value |= (1 << input_bit)
        matrix.append(row_value)

    return matrix


def gf2_gauss_solve(matrix: list, target: int) -> Optional[int]:
    """Solve M * x = target in GF(2) by Gaussian elimination."""
    # Augmented [M | target]
    aug_matrix = []
    for i in range(32):
        target_bit = (target >> i) & 1
        aug_matrix.append([matrix[i], target_bit])

    # Forward elimination
    for col in range(32):
        pivot_row = None
        for row in range(col, 32):
            if (aug_matrix[row][0] >> col) & 1:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        if pivot_row != col:
            aug_matrix[col], aug_matrix[pivot_row] = aug_matrix[pivot_row], aug_matrix[col]

        for row in range(32):
            if row != col and ((aug_matrix[row][0] >> col) & 1):
                aug_matrix[row][0] ^= aug_matrix[col][0]
                aug_matrix[row][1] ^= aug_matrix[col][1]

    # Back substitution
    solution = 0
    for row in range(32):
        row_matrix = aug_matrix[row][0]
        target_bit = aug_matrix[row][1]

        if row_matrix == 0:
            if target_bit != 0:
                return None
            continue

        for col in range(32):
            if (row_matrix >> col) & 1:
                if target_bit:
                    solution |= (1 << col)
                break

    return solution


def solve_crc32_patch_matrix(data: bytes, start_offset: int, end_offset: int,
                             patch_offset: int, initial_value: int, target_crc: int) -> Optional[int]:
    """Solve the patch dword that forces CRC32 over the region to target_crc.

    Exploits CRC linearity over GF(2) — no iterative search.
    """
    if patch_offset < start_offset or patch_offset + 3 > end_offset:
        return None

    def calc_crc_range(data_bytes: bytes, start: int, end_incl: int, init_val: int) -> int:
        crc = init_val
        pos = start
        while pos + 3 <= end_incl:
            dword = struct.unpack('<I', data_bytes[pos:pos+4])[0]
            pos += 4
            for _ in range(32):
                xor_result = dword ^ crc
                dword >>= 1
                if (xor_result & 1) != 0:
                    crc = (crc >> 1) ^ 0xEDB88320
                else:
                    crc = crc >> 1
        return crc

    intermediate_crc = calc_crc_range(data, start_offset, patch_offset - 1, initial_value)

    matrix = build_crc_transformation_matrix(intermediate_crc)

    data_copy = bytearray(data)
    struct.pack_into('<I', data_copy, patch_offset, 0x00000000)
    crc_with_zero = calc_crc_range(bytes(data_copy), start_offset, end_offset, initial_value)

    # matrix * patch = target ^ crc_with_zero
    target_diff = target_crc ^ crc_with_zero
    patch_value = gf2_gauss_solve(matrix, target_diff)

    return patch_value


@dataclass
class ChecksumStructure:
    """32-byte checksum structure within a Bosch block"""
    offset: int
    cs_block_id: int
    cs_start: int
    cs_end: int
    cs_start_val: int  # Often 0xFADECAFE
    cs_expected_val: int  # Often 0xCAFEAFFE
    block_id_ref: int
    block_id_addr: int
    cs_algorithm: int  # 0x00=CRC32, 0x01=ADD32, 0x10=ADD16
    calculated_checksum: Optional[int] = None
    is_valid: Optional[bool] = None


@dataclass
class CVNConfig:
    """CVN (Calibration Verification Number) configuration"""
    config_offset: int
    regions: List[tuple]  # (start, end) memory addresses
    ds_start: int
    ds_wocs_end: int  # dataset WOCS (without checksum) end
    base_address: int  # memory -> file offset base
    calculated_cvn: Optional[int] = None


@dataclass
class VariantTable:
    """One variant's copy of the parameter address table."""
    start: int          # file offset
    end: int            # file offset, exclusive
    match_pct: int


@dataclass
class VariantConfig:
    """Everything needed to correct the per-variant monitoring checksums."""
    block: 'BoschBlock'
    count: int                      # main + overrides, from the block header
    base_start: int                 # base address table, file offsets
    base_end: int
    region_start: int               # monitored region, file offsets
    region_end: int                 # exclusive; the compensation dword sits here
    adjust_offset: int
    seed: int
    expected: int
    tables: List[VariantTable] = field(default_factory=list)


@dataclass
class BoschBlock:
    """Represents a Bosch checksum block"""
    bin_start: int
    bin_end: int
    block_start: int
    block_end: int
    block_identifier: int
    block_name: str
    size: int
    sw_identifier: bytes
    num_checksum_structures: int
    checksum_adjust: int
    checksum: int
    checksum_complement: int
    checksum_structures: List[ChecksumStructure]

    @property
    def block_type_id(self) -> int:
        return self.block_identifier & 0xFF

    @property
    def has_otp(self) -> bool:
        """OTP (one-time programmable) flag."""
        return bool(self.block_identifier & 0x00800000)

    @property
    def is_unprogrammed(self) -> bool:
        """Reserved partition — checksum and adjust are both blank fill.

        Nothing real to protect (e.g. an unused second application slot), so
        correction must leave it alone rather than fabricate a checksum.
        """
        return (self.checksum in MEDC17_BLANK_WORDS and
                self.checksum_adjust in MEDC17_BLANK_WORDS)


class MEDC17BinaryParser:
    """Parser for MEDC17 ECU binary files (little-endian format)"""

    CHECKSUM_STRUCTURE_SIZE = 32

    ALGORITHM_NAMES = {0x00: 'CRC32', 0x01: 'ADD32', 0x10: 'ADD16'}

    def __init__(self, binary_path: str):
        self.binary_path = Path(binary_path)
        self.data: bytes = b''
        self.bosch_blocks: List[BoschBlock] = []
        self.cvn_config: Optional[CVNConfig] = None
        self.variant_config: Optional[VariantConfig] = None

    def load_binary(self) -> None:
        if not self.binary_path.exists():
            raise FileNotFoundError(f"Binary file not found: {self.binary_path}")

        with open(self.binary_path, 'rb') as f:
            self.data = f.read()

        print_success(f"Loaded: {self.binary_path.name}")
        print_info(f"Size: 0x{len(self.data):X} ({len(self.data):,} bytes)")

    def read_dword_le(self, offset: int) -> int:
        if offset + 4 > len(self.data):
            return 0
        return struct.unpack('<I', self.data[offset:offset+4])[0]

    def read_word_le(self, offset: int) -> int:
        if offset + 2 > len(self.data):
            return 0
        return struct.unpack('<H', self.data[offset:offset+2])[0]

    def read_byte(self, offset: int) -> int:
        if offset >= len(self.data):
            return 0
        return self.data[offset]

    def read_checksum_structures(self, offset: int, count: int) -> List[ChecksumStructure]:
        structures = []

        for i in range(count):
            struct_offset = offset + (i * self.CHECKSUM_STRUCTURE_SIZE)
            if struct_offset + self.CHECKSUM_STRUCTURE_SIZE > len(self.data):
                break

            structure_data = self.data[struct_offset:struct_offset + self.CHECKSUM_STRUCTURE_SIZE]

            structures.append(ChecksumStructure(
                offset=struct_offset,
                cs_block_id=structure_data[0],
                cs_start=self.read_dword_le(struct_offset + 4),
                cs_end=self.read_dword_le(struct_offset + 8),
                cs_start_val=self.read_dword_le(struct_offset + 12),
                cs_expected_val=self.read_dword_le(struct_offset + 16),
                block_id_ref=self.read_dword_le(struct_offset + 20),
                block_id_addr=self.read_dword_le(struct_offset + 24),
                cs_algorithm=self.read_word_le(struct_offset + 28) & 0xFF,
            ))

        return structures

    def parse_block(self, flat_address: int) -> Optional[BoschBlock]:
        """Parse the block at flat_address, or None if it doesn't validate.

        Block structure (little-endian):
        +0x00: Block identifier (dword)
        +0x04: Size (dword)
        +0x0C: Block end address (dword)
        +0x1A: Software identifier (10 bytes)
        +0x2C: Number of checksum structures (dword)
        +0x30: Checksum adjust (dword)
        +0x34: Checksum structures start (32 bytes each)
        Last: Final checksum (dword)
        """
        if flat_address + 0x40 > len(self.data):
            return None

        block_identifier = self.read_dword_le(flat_address)
        size = self.read_dword_le(flat_address + 4)
        block_end = self.read_dword_le(flat_address + 12)

        block_type_id = block_identifier & 0xFF
        if block_type_id not in BLOCK_IDENTIFIERS:
            return None

        # 0x40 = minimum header size
        if size < 0x40 or size > len(self.data) or flat_address + size > len(self.data):
            return None

        # Blocks terminate with DEADBEEF
        block_end_offset = flat_address + size - 4
        if self.read_dword_le(block_end_offset) != 0xDEADBEEF:
            return None

        if not self._is_flash_addr(block_end):
            return None

        # +0x1A
        identifier_length = 10
        sw_identifier = self.data[flat_address + 26:flat_address + 26 + identifier_length]

        # +0x2C
        num_checksum_structures = self.read_dword_le(flat_address + 26 + identifier_length + 8)

        # The DLL caps this at 8
        if not 1 <= num_checksum_structures <= 8:
            return None

        block_start = ((block_end + 5) - size - 1)

        if not self._is_flash_addr(block_start):
            return None

        checksum_adjust = self.read_dword_le(flat_address + 0x30)

        checksum_structures = self.read_checksum_structures(
            flat_address + 0x34,
            num_checksum_structures
        )

        checksum_offset = flat_address + 0x34 + (num_checksum_structures * self.CHECKSUM_STRUCTURE_SIZE)
        checksum = self.read_dword_le(checksum_offset)
        checksum_complement = (~checksum) & 0xFFFFFFFF

        block_name = BLOCK_IDENTIFIERS.get(block_type_id, f'Unknown (0x{block_type_id:02X})')

        return BoschBlock(
            bin_start=flat_address,
            bin_end=flat_address + size - 1,
            block_start=block_start,
            block_end=block_end + 3,
            block_identifier=block_identifier,
            block_name=block_name,
            size=size,
            sw_identifier=sw_identifier,
            num_checksum_structures=num_checksum_structures,
            checksum_adjust=checksum_adjust,
            checksum=checksum,
            checksum_complement=checksum_complement,
            checksum_structures=checksum_structures,
        )

    def _has_block_signature(self, hdr: int) -> bool:
        """Confirm a magic-pair hit is a real block start.

        Checks the fixed fields of the 72-byte descriptor, which rejects interior
        checksum structures (they carry the same magic pair) and coincidences:
          +0x00  block_type   a known identifier
          +0x01  reserved     0x00
          +0x03  reserved     0x00
          +0x0F  block_class  0x80 (application) or 0xA0 (data/cal)
        The caller already guarantees the magic pair at +0x40/+0x44.
        """
        if hdr < 0 or hdr + 72 > len(self.data):
            return False
        d = self.data
        return (d[hdr] in BLOCK_IDENTIFIERS and
                d[hdr + 1] == 0x00 and d[hdr + 3] == 0x00 and
                d[hdr + 15] in MEDC17_BLOCK_CLASSES)

    def find_bosch_blocks(self) -> None:
        """Locate all Bosch checksum blocks via their FADECAFE/CAFEAFFE signature.

        Jumps to each occurrence of the magic pair (C-level substring search),
        backs off 64 bytes to the block start, confirms the header, then parses.
        Blocks are located independently, so a malformed one can't hide those
        after it.
        """
        console.print("\n[*] Scanning for Bosch checksum blocks...")

        self.bosch_blocks = []

        data = self.data
        search = 0
        while True:
            pos = data.find(MEDC17_MAGIC_PAIR, search)
            if pos < 0:
                break
            search = pos + 1                        # keep overlapping matches
            hdr = pos - MEDC17_MAGIC_OFFSET

            if not self._has_block_signature(hdr):
                continue

            block = self.parse_block(hdr)
            if block is None:
                continue

            self.bosch_blocks.append(block)
            console.print(f"[+] Found block {len(self.bosch_blocks)} at 0x{hdr:X}: {block.block_name}")

        # find() runs left-to-right so these are already in order; sort defensively
        self.bosch_blocks.sort(key=lambda b: b.bin_start)

        if not self.bosch_blocks:
            console.print("[!] No Bosch checksum blocks found")
            return

        console.print(f"[+] Total Bosch blocks found: {len(self.bosch_blocks)}")

    def identify_ecu_variant(self) -> List[str]:
        """ECU variant string from the Dataset #0 block. (This could probably be done a smarter way)

        Lives at +0x78 in the block (ID 0x60), as slash-separated fields, one of
        which holds the variant: "34/1/EDC17_C46/5/P643//C643X5L8///"
        """
        dataset_block = None
        for block in self.bosch_blocks:
            if block.block_identifier == 0x60:
                dataset_block = block
                break

        if not dataset_block:
            return ["Unknown (no Dataset block found)"]

        variant_offset = dataset_block.bin_start + 0x78

        if variant_offset + 100 > len(self.data):
            return ["Unknown (offset out of range)"]

        variant_data = self.data[variant_offset:variant_offset+100]

        null_pos = variant_data.find(b'\x00')
        if null_pos != -1:
            variant_data = variant_data[:null_pos]

        try:
            variant_string = variant_data.decode('ascii', errors='ignore')
        except:
            return ["Unknown (decode error)"]

        fields = variant_string.split('/')

        ecu_variant = None
        for field in fields:
            field = field.strip()
            if 'MED17' in field or 'EDC17' in field or 'MEDC17' in field:
                ecu_variant = field
                break

        if ecu_variant:
            return [ecu_variant]
        else:
            # Fallback: show the raw fields
            non_empty = [f for f in fields if f.strip()]
            if non_empty:
                return [f"Unknown variant (fields: {', '.join(non_empty[:3])})"]
            else:
                return ["Unknown"]

    def calculate_crc32_algo(self, start: int, end_inclusive: int, initial_value: int) -> int:
        """CRC32 checksum, algorithm 0x00 (SB_CRC32_ALGO_E).

        Bit-by-bit CRC32-IEEE over little-endian dwords, matching the TriCore
        implementation exactly — a table version would be faster but is not
        guaranteed to agree. A pass gives 0x35015001, the complement of
        0xCAFEAFFE. initial_value is usually 0xFADECAFE.
        """
        if start < 0 or end_inclusive >= len(self.data) or start > end_inclusive:
            return 0

        crc = initial_value
        pos = start

        while pos + 3 <= end_inclusive:
            dword = struct.unpack('<I', self.data[pos:pos+4])[0]
            pos += 4

            for _ in range(32):
                xor_result = dword ^ crc
                dword >>= 1
                if (xor_result & 1) != 0:
                    crc = (crc >> 1) ^ 0xEDB88320
                else:
                    crc = crc >> 1

        return crc

    def calculate_add32_checksum(self, start: int, end_inclusive: int, initial_value: int) -> int:
        """ADD32 checksum, algorithm 0x01 (SB_ADD32_ALGO_E): sum of dwords."""
        if start < 0 or end_inclusive >= len(self.data) or start > end_inclusive:
            return 0

        checksum = initial_value
        pos = start

        while pos + 3 <= end_inclusive:
            dword = struct.unpack('<I', self.data[pos:pos+4])[0]
            pos += 4
            checksum = (checksum + dword) & 0xFFFFFFFF

        return checksum

    def calculate_add16_checksum(self, start: int, end_inclusive: int, initial_value: int) -> int:
        """ADD16 checksum, algorithm 0x10 (SB_ADD16_ALGO_E): sum of 16-bit words.

        The main loop folds each dword the way the DLL does:

            lc = *startAdr++;
            chkSum_u32 += (uint16)lc + (uint16)(lc >> 16);

        The final word is the exception — it lands in the high half instead of
        being folded down. That makes the last dword of the region contribute its
        full 32-bit value (low word flat, high word shifted), which is what turns
        it into the adjust slot correct_add16_checksum writes to: the checksum
        then moves by exactly the value added there.

        Do not "simplify" the tail to a flat add. Verified against factory
        firmware (07P906027A): flat gives 0x4291386C where the ECU expects
        0xCAFEAFFE, shifted reproduces the stored value exactly.
        """
        if start < 0 or end_inclusive >= len(self.data) or start > end_inclusive:
            return 0

        checksum = initial_value
        pos = start

        while pos <= end_inclusive - 2:
            word = self.data[pos] | (self.data[pos + 1] << 8)
            pos += 2
            checksum = (checksum + word) & 0xFFFFFFFF

        # Last 16-bit word goes into the high 16 bits
        word = self.data[pos] | (self.data[pos + 1] << 8)
        checksum = (checksum + (word << 16)) & 0xFFFFFFFF

        return checksum

    def _mem_to_file(self, mem_addr: int, block_start_mem: int, block_start_bin: int) -> int:
        """Flash memory address -> file offset, relative to a block.

        Treats the 0x8 and 0xA segment aliases as the same physical flash, so the
        base holds whichever view the block and the address use.
        """
        return (flash_canonical(mem_addr)
                - flash_canonical(block_start_mem)
                + block_start_bin)

    def validate_checksum_structure(self, cs: ChecksumStructure, block_start_mem: int,
                                     block_start_bin: int) -> bool:
        """Validate a checksum structure by recalculating over its region."""
        start_offset = self._mem_to_file(cs.cs_start, block_start_mem, block_start_bin)
        end_offset = self._mem_to_file(cs.cs_end, block_start_mem, block_start_bin)

        if start_offset < 0 or end_offset > len(self.data) or start_offset >= end_offset:
            cs.calculated_checksum = None
            cs.is_valid = False
            return False

        if cs.cs_algorithm == 0x00:
            # CRC32 compares against the complement; ADD32/ADD16 compare directly
            checksum = self.calculate_crc32_algo(start_offset, end_offset, cs.cs_start_val)
            cs.calculated_checksum = checksum
            cs.is_valid = (checksum == (~cs.cs_expected_val) & 0xFFFFFFFF)
        elif cs.cs_algorithm == 0x01:
            checksum = self.calculate_add32_checksum(start_offset, end_offset, cs.cs_start_val)
            cs.calculated_checksum = checksum
            cs.is_valid = (checksum == cs.cs_expected_val)
        elif cs.cs_algorithm == 0x10:
            checksum = self.calculate_add16_checksum(start_offset, end_offset, cs.cs_start_val)
            cs.calculated_checksum = checksum
            cs.is_valid = (checksum == cs.cs_expected_val)
        else:
            cs.calculated_checksum = None
            cs.is_valid = None

        return cs.is_valid if cs.is_valid is not None else False

    def validate_all_checksums(self) -> None:
        console.print("\n[*] Validating checksums...")

        for block in self.bosch_blocks:
            for cs in block.checksum_structures:
                self.validate_checksum_structure(cs, block.block_start, block.bin_start)

        counts = self.checksum_counts()

        if counts['reserved_checksums']:
            console.print(f"[+] Validated {counts['valid_checksums']}/{counts['checkable_checksums']} "
                          f"checksums ({counts['reserved_checksums']} reserved/unprogrammed skipped)")
        else:
            console.print(f"[+] Validated {counts['valid_checksums']}/{counts['total_checksums']} checksums")

    def checksum_counts(self) -> dict:
        """Checksum tally across every block.

        Reserved partitions have no finalised checksum, so they are counted
        separately rather than as failures.
        """
        total = reserved = valid = invalid = unknown = 0

        for block in self.bosch_blocks:
            for cs in block.checksum_structures:
                total += 1
                if block.is_unprogrammed:
                    reserved += 1
                elif cs.is_valid is None:
                    unknown += 1
                elif cs.is_valid:
                    valid += 1
                else:
                    invalid += 1

        return {
            'total_checksums': total,
            'checkable_checksums': total - reserved,
            'valid_checksums': valid,
            'invalid_checksums': invalid,
            'unknown_checksums': unknown,
            'reserved_checksums': reserved,
            'all_valid': invalid == 0,
        }

    def _is_flash_addr(self, addr: int) -> bool:
        """Cached (0x8xxxxxxx) or uncached (0xAxxxxxxx) TriCore flash address.

        Data/calibration blocks (block_class 0xA0) use the uncached alias.
        """
        return (0x80000000 <= addr <= 0x8FFFFFFF or
                0xA0000000 <= addr <= 0xAFFFFFFF)

    # Variant dataset --------------------------------------------------------

    def _image_offset(self, mem_addr: int, data: bytes = None) -> Optional[int]:
        """Flash address to file offset, image-wide, anchored on the lowest block."""
        if not self.bosch_blocks:
            return None
        anchor = self.bosch_blocks[0]
        offset = (flash_canonical(mem_addr)
                  - flash_canonical(anchor.block_start)
                  + anchor.bin_start)
        limit = len(data if data is not None else self.data)
        return offset if 0 <= offset < limit else None

    def _entry_offset(self, value: int, data: bytes) -> Optional[int]:
        """A table entry translated to a file offset, or None if unusable."""
        if value == 0xFFFFFFFF:
            return None
        return self._image_offset(value, data)

    def _find_monitored_region(self, data: bytes) -> Optional[tuple]:
        """Locate the structure carrying the monitoring checksum.

        Returns (region_start, region_end_exclusive, adjust_offset, seed, expected).
        The structure is identified by its marker bytes; the last dword of its
        range is the compensation slot and is excluded from the sum.
        """
        for block in self.bosch_blocks:
            for cs in block.checksum_structures:
                if data[cs.offset:cs.offset + 4] != VARIANT_MONITOR_MARKER:
                    continue
                start = self._image_offset(cs.cs_start, data)
                end = self._image_offset(cs.cs_end, data)
                if start is None or end is None or end - 4 <= start:
                    continue
                if struct.unpack_from('<I', data, end - 3)[0] in MEDC17_BLANK_WORDS:
                    continue
                return start, end - 4, end - 3, cs.cs_start_val, cs.cs_expected_val
        return None

    def _find_base_table(self, data: bytes, exclude: tuple = None) -> Optional[tuple]:
        """Find the base parameter address table.

        An address table is a long strictly-ascending run of flash addresses.
        Returns (start_offset, end_offset_exclusive).
        """
        limit = len(data) - 4
        run_start = None
        prev = 0
        pos = 0
        while pos < limit:
            if exclude and exclude[0] <= pos <= exclude[1]:
                pos += 4
                continue
            value = struct.unpack_from('<I', data, pos)[0]
            ascending = value > prev and self._is_flash_addr(value)
            if ascending:
                if run_start is None:
                    run_start = pos
                prev = value
            else:
                if run_start is not None and (pos - run_start) // 4 >= VARIANT_TABLE_MIN_RUN:
                    first = struct.unpack_from('<I', data, run_start)[0]
                    last = struct.unpack_from('<I', data, pos - 4)[0]
                    if last - first >= VARIANT_TABLE_MIN_SPAN:
                        return run_start, pos
                run_start = None
                prev = 0
            pos += 4
        return None

    def _find_variant_tables(self, data: bytes, base_start: int,
                             base_end: int) -> List[VariantTable]:
        """Find copies of the base table — one per variant.

        Variants share most of their entries with the base and differ only where
        the variant overrides a parameter, so copies are matched by similarity
        rather than equality.
        """
        length = base_end - base_start
        words = length // 4
        if words == 0:
            return []
        first = struct.unpack_from('<I', data, base_start)[0]
        base_words = struct.unpack_from(f'<{words}I', data, base_start)

        tables: List[VariantTable] = []
        pos = base_end
        while pos + length <= len(data):
            if struct.unpack_from('<I', data, pos)[0] != first:
                pos += 4
                continue
            other = struct.unpack_from(f'<{words}I', data, pos)
            matched = sum(1 for a, b in zip(base_words, other) if a == b)
            pct = (matched * 100) // words
            if pct >= VARIANT_TABLE_MIN_MATCH:
                tables.append(VariantTable(pos, pos + length, pct))
                if len(tables) >= VARIANT_MAX_TABLES:
                    break
                pos += length
                continue
            pos += 4
        return tables

    def _table_offsets(self, data: bytes, start: int, length: int) -> tuple:
        """Table entries as file offsets, plus the index where a walk would abort."""
        words = length // 4
        raw = struct.unpack_from(f'<{words}I', data, start)
        offsets = [self._entry_offset(v, data) for v in raw]
        stop = words
        for i, (value, offset) in enumerate(zip(raw, offsets)):
            if offset is None or value == VARIANT_REMAP_SENTINEL:
                stop = i
                break
        return offsets, stop

    def _build_remap(self, data: bytes, cfg: VariantConfig,
                     table: VariantTable) -> tuple:
        """Precompute the base/variant offset pair used to translate addresses."""
        length = cfg.base_end - cfg.base_start
        base_offsets, base_stop = self._table_offsets(data, cfg.base_start, length)
        var_offsets, var_stop = self._table_offsets(data, table.start, length)
        return base_offsets, var_offsets, min(base_stop, var_stop)

    def _remap_offset(self, offset: int, remap: tuple) -> Optional[int]:
        """Translate a file offset from the base table's layout into a variant's.

        Entries are slot boundaries: the offset is located between two adjacent
        base entries and re-based onto the matching slot of the variant table,
        keeping its position within the slot.
        """
        base_offsets, var_offsets, stop = remap
        index = bisect.bisect_right(base_offsets, offset, 0, stop) - 1
        if index < 0 or index + 1 >= stop:
            return None
        low, high = base_offsets[index], base_offsets[index + 1]
        if not (low <= offset < high - 1):
            return None
        return var_offsets[index] - low + offset

    def _variant_sum(self, data: bytes, cfg: VariantConfig,
                     remap: Optional[tuple]) -> int:
        """Sum the monitored region as the given variant sees it.

        Reads resolve through the variant's table, so an overridden parameter
        contributes the variant's own copy rather than the shared one.
        """
        total = cfg.seed
        for addr in range(cfg.region_start, cfg.region_end, 4):
            word = 0
            for half in (0, 2):
                src = addr + half
                if remap is not None:
                    mapped = self._remap_offset(src, remap)
                    if mapped is not None and 0 <= mapped + 2 <= len(data):
                        src = mapped
                word |= struct.unpack_from('<H', data, src)[0] << (half * 8)
            total = (total + word) & 0xFFFFFFFF
        return total

    def find_variant_config(self, data: bytes = None) -> Optional[VariantConfig]:
        """Detect a multi-variant calibration and everything needed to correct it."""
        data = self.data if data is None else data

        block = next((b for b in self.bosch_blocks
                      if b.block_type_id == VARIANT_BLOCK_ID), None)
        if block is None:
            return None

        count = self.read_dword_le(block.bin_start + VARIANT_COUNT_OFFSET)
        if count <= 1 or count > VARIANT_MAX_TABLES + 1:
            return None

        region = self._find_monitored_region(data)
        if region is None:
            return None

        base = self._find_base_table(data, exclude=(block.bin_start, block.bin_end))
        if base is None:
            base = self._find_base_table(data)
        if base is None:
            return None

        cfg = VariantConfig(
            block=block,
            count=count,
            base_start=base[0],
            base_end=base[1],
            region_start=region[0],
            region_end=region[1],
            adjust_offset=region[2],
            seed=region[3],
            expected=region[4],
        )
        cfg.tables = self._find_variant_tables(data, cfg.base_start, cfg.base_end)
        self.variant_config = cfg
        return cfg

    def correct_variant_checksums(self, data: bytearray) -> int:
        """Write a compensation dword for every variant. Returns the number written.

        Must run before the CRC32 pass — the values land inside checksummed
        blocks.
        """
        cfg = self.find_variant_config(bytes(data))
        if cfg is None:
            return 0

        if not cfg.tables:
            print_warning("Variant dataset present but no variant tables found "
                          "— per-variant checksums not corrected")
            return 0

        written = 0
        for index, table in enumerate([None] + cfg.tables):
            remap = None
            if table is None:
                target = cfg.adjust_offset
                label = "main"
            else:
                remap = self._build_remap(bytes(data), cfg, table)
                target = self._remap_offset(cfg.adjust_offset, remap)
                label = f"variant {index}"
                if target is None:
                    print_warning(f"{label}: compensation address not in table — skipped")
                    continue
            if target + 4 > len(data):
                print_warning(f"{label}: compensation address outside binary — skipped")
                continue

            total = self._variant_sum(bytes(data), cfg, remap)
            adjust = (cfg.expected - total) & 0xFFFFFFFF
            current = struct.unpack_from('<I', data, target)[0]
            if current == adjust:
                console.print(f"  {label}: [green]✓[/green] Already valid")
                continue
            struct.pack_into('<I', data, target, adjust)
            console.print(f"  {label}: 0x{current:08X} → [cyan]0x{adjust:08X}[/cyan] "
                          f"at 0x{target:08X}")
            written += 1
        return written

    def variant_to_dict(self) -> Optional[dict]:
        cfg = self.variant_config or self.find_variant_config()
        if cfg is None:
            return None
        return {
            'variants': cfg.count,
            'tables_found': len(cfg.tables),
            'base_table': {'start': hex32(cfg.base_start), 'end': hex32(cfg.base_end)},
            'monitored_region': {'start': hex32(cfg.region_start),
                                 'end': hex32(cfg.region_end),
                                 'adjust': hex32(cfg.adjust_offset)},
            'similarity': [t.match_pct for t in cfg.tables],
        }

    def find_cvn_config(self) -> Optional[CVNConfig]:
        """Find and parse the CVN config: pointers to the regions the CVN covers."""
        ds_block = None
        for block in self.bosch_blocks:
            if block.block_identifier == 0x60:
                ds_block = block
                break

        if not ds_block:
            return None

        ds_start = ds_block.block_start
        ds_end = ds_block.block_end
        # Canonicalise so pointers in either segment translate against one base
        base = flash_canonical(self.bosch_blocks[0].block_start) - self.bosch_blocks[0].bin_start

        # Config pattern: { pointer, DS_START, DS_WOCS_END, count }
        for offset in range(0, len(self.data) - 16, 4):
            ptr = self.read_dword_le(offset)
            val1 = self.read_dword_le(offset + 4)
            val2 = self.read_dword_le(offset + 8)
            count = self.read_dword_le(offset + 12)

            if (self._is_flash_addr(ptr) and val1 == ds_start and
                self._is_flash_addr(val2) and ds_start < val2 <= ds_end and
                1 <= count <= 4):

                # Follow the pointer to the memory section table
                config_offset = flash_canonical(ptr) - base
                if not (0 <= config_offset < len(self.data)):
                    continue

                memsec_ptr = self.read_dword_le(config_offset)
                memsec_offset = flash_canonical(memsec_ptr) - base
                if not (0 <= memsec_offset < len(self.data)):
                    continue

                # (start, end) pairs, canonicalised so every downstream
                # `addr - base` holds whichever segment the firmware used
                regions = []
                for i in range(4):  # max 4 sections
                    sec_start = self.read_dword_le(memsec_offset + i * 8)
                    sec_end = self.read_dword_le(memsec_offset + i * 8 + 4)
                    if self._is_flash_addr(sec_start) and self._is_flash_addr(sec_end) and sec_end > sec_start:
                        regions.append((flash_canonical(sec_start), flash_canonical(sec_end)))
                    else:
                        break

                # Dataset region goes last — _dataset_comptest_bounds relies on it
                regions.append((flash_canonical(val1), flash_canonical(val2)))

                return CVNConfig(
                    config_offset=offset,
                    regions=regions,
                    ds_start=flash_canonical(val1),
                    ds_wocs_end=flash_canonical(val2),
                    base_address=base
                )

        return None

    def calculate_cvn(self, data: bytes = None) -> Optional[int]:
        """CVN (Calibration Verification Number): CRC32 over the config's regions.

        Table-driven over little-endian dwords, matching the TriCore original.
        """
        if self.cvn_config is None:
            return None

        if data is None:
            data = self.data

        init_crc32_table()

        crc = 0xFFFFFFFF
        base = self.cvn_config.base_address

        for mem_start, mem_end in self.cvn_config.regions:
            file_start = mem_start - base
            file_end = mem_end - base

            if file_start < 0 or file_end > len(data):
                continue

            # Table-driven dword at a time — matches the bit-by-bit version, ~8x faster
            pos = file_start
            while pos + 3 <= file_end:
                crc = CRC32_TABLE[(crc ^ data[pos]) & 0xFF] ^ (crc >> 8)
                crc = CRC32_TABLE[(crc ^ data[pos+1]) & 0xFF] ^ (crc >> 8)
                crc = CRC32_TABLE[(crc ^ data[pos+2]) & 0xFF] ^ (crc >> 8)
                crc = CRC32_TABLE[(crc ^ data[pos+3]) & 0xFF] ^ (crc >> 8)
                pos += 4

        return crc ^ 0xFFFFFFFF

    def _dataset_comptest_bounds(self):
        """File offsets of the dataset CompTest region, or None without a CVN config.

        The dataset region (the last CVN region) is what the ECU CompTest CRC
        covers, and the stored CRC sits in the 4 bytes right after it. Returns
        (region_start, region_end_incl, blk_end); the CRC is [blk_end-3, blk_end].
        """
        if self.cvn_config is None:
            return None
        base = self.cvn_config.base_address
        region_start = self.cvn_config.ds_start - base
        region_end_incl = self.cvn_config.ds_wocs_end - base
        blk_end = region_end_incl + 4
        return region_start, region_end_incl, blk_end

    def find_erased_slot(self, data, blk_end: int, region_start: int) -> Optional[int]:
        """First free/erased dword that can host a compensation value.

        Walks back a dword at a time from one paragraph below the stored CRC,
        (blk_end & ~0xF) - 0x10, bounded by region_start so the slot stays inside
        the checksummed region. Returns a file offset, or None.
        """
        off = (blk_end & 0xFFFFFFF0) - 0x10
        while off > region_start:
            if off + 4 <= len(data):
                value = struct.unpack('<I', data[off:off + 4])[0]
                if value in ERASED_DWORDS:
                    return off
            off -= 4
        return None

    def correct_cvn(self, target_cvn: int, data: bytearray) -> bool:
        """Patch the dataset region so the CVN equals target_cvn.

        Solves the patch dword in GF(2) and writes it to a free/erased slot, so
        live calibration data is never overwritten.
        """
        if self.cvn_config is None:
            return False

        region_start, _region_end_incl, blk_end = self._dataset_comptest_bounds()

        # An erased slot, not DS_WOCS_END & ~0x1F, which could land on live data
        patch_offset = self.find_erased_slot(data, blk_end, region_start)
        if patch_offset is None:
            print_error("No erased slot found in dataset region for CVN patch")
            return False

        if not self._offset_in_cvn_region(patch_offset):
            print_error("CVN patch location not within any CVN region")
            return False

        return self._correct_cvn_multiregion(target_cvn, data, patch_offset)

    def correct_cvn_best_effort(self, data: bytearray) -> bool:
        """Preserve the CVN without the original file.

        The ECU's CompTest CRC covers the same dataset region as the CVN and is
        stored in the 4 bytes right after it. Both are CRC32 over that region and
        CRC32 is affine, so a compensation dword that returns CompTest to its
        stored value restores the untuned CVN with it — no original needed.

        Best effort: assumes the tune didn't overwrite the stored CompTest CRC.
        """
        if self.cvn_config is None:
            print_error("No CVN configuration found")
            return False

        region_start, region_end_incl, blk_end = self._dataset_comptest_bounds()

        # Stored CRC: the dword immediately after the dataset region
        if blk_end + 1 > len(data):
            print_error("Dataset CompTest CRC lies outside the binary")
            return False
        stored_crc = struct.unpack('<I', data[blk_end - 3:blk_end + 1])[0]
        print_info(f"Stored CompTest CRC: 0x{stored_crc:08X}")

        if stored_crc in ERASED_DWORDS:
            print_warning("Stored CompTest CRC looks blank/erased — "
                          "this dataset may not carry a CompTest checksum")

        current_crc = self._region_crc32(data, region_start, region_end_incl)
        print_info(f"Current CompTest CRC: 0x{current_crc:08X}")
        if current_crc == stored_crc:
            print_success("CompTest CRC already matches — CVN already preserved")
            return True

        slot = self.find_erased_slot(data, blk_end, region_start)
        if slot is None:
            print_error("No erased slot found in dataset region for compensation")
            return False
        print_info(f"Compensation slot: 0x{slot:08X} (was 0x{struct.unpack('<I', data[slot:slot+4])[0]:08X})")

        patch = crc32_fold_patch(data, region_start, region_end_incl, slot,
                                 init=0xFFFFFFFF, target=stored_crc, xorout=0xFFFFFFFF)
        struct.pack_into('<I', data, slot, patch)

        new_crc = self._region_crc32(data, region_start, region_end_incl)
        if new_crc == stored_crc:
            print_success(f"CompTest CRC restored to 0x{stored_crc:08X} "
                          f"(compensation 0x{patch:08X} @ 0x{slot:08X})")
            return True

        print_error(f"CompTest CRC still 0x{new_crc:08X}, expected 0x{stored_crc:08X}")
        return False

    @staticmethod
    def _region_crc32(data, start: int, end_incl: int) -> int:
        """Standard CRC32 (zlib) over an inclusive byte range — the CompTest CRC."""
        import zlib
        return zlib.crc32(bytes(data[start:end_incl + 1])) & 0xFFFFFFFF

    def _offset_in_cvn_region(self, offset: int) -> bool:
        """Whether a file offset falls inside one of the CVN's covered regions."""
        if self.cvn_config is None:
            return False

        base = self.cvn_config.base_address
        for mem_start, mem_end in self.cvn_config.regions:
            if (mem_start - base) <= offset < (mem_end - base):
                return True

        return False

    def cvn_feasibility(self, data=None) -> dict:
        """Report the CVN, and whether it can be preserved, without touching the file.

        Checks the same conditions correct_cvn_best_effort relies on, so a caller
        can offer CVN preservation only when it stands a chance of succeeding.
        """
        if data is None:
            data = self.data

        status = {
            'available': self.cvn_config is not None,
            'value': None,
            'regions': 0,
            'regions_in_file': 0,
            'stored_comptest_crc': None,
            'current_comptest_crc': None,
            'comptest_matches': None,
            'compensation_slot': None,
            'preserve_supported': False,
            'preserve_reason': None,
            'match_original_supported': False,
        }

        if self.cvn_config is None:
            status['preserve_reason'] = 'No CVN configuration found in this binary'
            return status

        base = self.cvn_config.base_address
        readable = sum(1 for mem_start, mem_end in self.cvn_config.regions
                       if 0 <= (mem_start - base) and (mem_end - base) <= len(data))

        # Recomputed from the data handed in, so the same call reports the CVN
        # before and after a correction
        status['value'] = hex32(self.calculate_cvn(bytes(data)))
        status['regions'] = len(self.cvn_config.regions)
        status['regions_in_file'] = readable

        if readable < len(self.cvn_config.regions):
            # A partial dump: the CVN covers flash this file doesn't contain, so
            # the calculated value means nothing and nothing can be preserved
            status['value'] = None
            status['preserve_reason'] = ('CVN regions extend past the end of this file — '
                                         'it looks like a partial dump')
            return status

        region_start, region_end_incl, blk_end = self._dataset_comptest_bounds()

        if blk_end + 1 > len(data) or region_start < 0:
            status['preserve_reason'] = 'Dataset CompTest CRC lies outside the binary'
            return status

        stored_crc = struct.unpack('<I', data[blk_end - 3:blk_end + 1])[0]
        current_crc = self._region_crc32(data, region_start, region_end_incl)
        slot = self.find_erased_slot(data, blk_end, region_start)

        status['stored_comptest_crc'] = hex32(stored_crc)
        status['current_comptest_crc'] = hex32(current_crc)
        status['comptest_matches'] = stored_crc == current_crc
        status['compensation_slot'] = hex32(slot) if slot is not None else None
        status['match_original_supported'] = slot is not None and self._offset_in_cvn_region(slot)

        if stored_crc in ERASED_DWORDS:
            status['preserve_reason'] = ('Stored CompTest CRC is blank — this dataset carries no '
                                         'CompTest checksum to restore the CVN from')
        elif stored_crc == current_crc:
            status['preserve_supported'] = True
            status['preserve_reason'] = 'CompTest CRC already matches — the CVN is unchanged'
        elif slot is None:
            status['preserve_reason'] = ('No erased slot in the dataset region to hold the '
                                         'compensation dword')
        else:
            status['preserve_supported'] = True

        return status

    def _correct_cvn_multiregion(self, target_cvn: int, data: bytearray, patch_offset: int) -> bool:
        """Solve the CVN patch dword when the patch sits in a multi-region CRC.

        GF(2) again, O(n + log(n)*32^3): the patch's local effect is propagated to
        the end of the data by matrix exponentiation, rather than recomputing the
        whole CVN 33 times.
        """
        init_crc32_table()
        base = self.cvn_config.base_address

        patch_region_idx = None
        for idx, (mem_start, mem_end) in enumerate(self.cvn_config.regions):
            file_start = mem_start - base
            file_end = mem_end - base
            if file_start <= patch_offset < file_end:
                patch_region_idx = idx
                break

        if patch_region_idx is None:
            return False

        patch_region_file_start = self.cvn_config.regions[patch_region_idx][0] - base
        patch_region_file_end = self.cvn_config.regions[patch_region_idx][1] - base

        data_copy = bytearray(data)
        struct.pack_into('<I', data_copy, patch_offset, 0)
        cvn_with_zero = self.calculate_cvn(bytes(data_copy))

        # CRC up to, but not including, the patch dword
        crc_to_patch = 0xFFFFFFFF

        for idx in range(patch_region_idx):
            mem_start, mem_end = self.cvn_config.regions[idx]
            file_start = mem_start - base
            file_end = mem_end - base
            if file_start < 0 or file_end > len(data):
                continue
            pos = file_start
            while pos + 3 <= file_end:
                crc_to_patch = CRC32_TABLE[(crc_to_patch ^ data[pos]) & 0xFF] ^ (crc_to_patch >> 8)
                crc_to_patch = CRC32_TABLE[(crc_to_patch ^ data[pos+1]) & 0xFF] ^ (crc_to_patch >> 8)
                crc_to_patch = CRC32_TABLE[(crc_to_patch ^ data[pos+2]) & 0xFF] ^ (crc_to_patch >> 8)
                crc_to_patch = CRC32_TABLE[(crc_to_patch ^ data[pos+3]) & 0xFF] ^ (crc_to_patch >> 8)
                pos += 4

        pos = patch_region_file_start
        while pos < patch_offset:
            crc_to_patch = CRC32_TABLE[(crc_to_patch ^ data[pos]) & 0xFF] ^ (crc_to_patch >> 8)
            crc_to_patch = CRC32_TABLE[(crc_to_patch ^ data[pos+1]) & 0xFF] ^ (crc_to_patch >> 8)
            crc_to_patch = CRC32_TABLE[(crc_to_patch ^ data[pos+2]) & 0xFF] ^ (crc_to_patch >> 8)
            crc_to_patch = CRC32_TABLE[(crc_to_patch ^ data[pos+3]) & 0xFF] ^ (crc_to_patch >> 8)
            pos += 4

        # Effect of each patch bit on the CRC immediately after the patch dword
        def process_dword_table(crc_in, b0, b1, b2, b3):
            crc = crc_in
            crc = CRC32_TABLE[(crc ^ b0) & 0xFF] ^ (crc >> 8)
            crc = CRC32_TABLE[(crc ^ b1) & 0xFF] ^ (crc >> 8)
            crc = CRC32_TABLE[(crc ^ b2) & 0xFF] ^ (crc >> 8)
            crc = CRC32_TABLE[(crc ^ b3) & 0xFF] ^ (crc >> 8)
            return crc

        baseline_crc = process_dword_table(crc_to_patch, 0, 0, 0, 0)
        patch_effects_local = []
        for bit in range(32):
            patch_val = 1 << bit
            b0 = patch_val & 0xFF
            b1 = (patch_val >> 8) & 0xFF
            b2 = (patch_val >> 16) & 0xFF
            b3 = (patch_val >> 24) & 0xFF
            test_crc = process_dword_table(crc_to_patch, b0, b1, b2, b3)
            patch_effects_local.append(test_crc ^ baseline_crc)

        # Bytes CRC'd after the patch, one table step each. Count whole dwords
        # only, matching the CVN loop's "while pos + 3 <= file_end".
        total_bytes_after = 0

        pos = patch_offset + 4
        while pos + 3 <= patch_region_file_end:
            total_bytes_after += 4
            pos += 4

        for idx in range(patch_region_idx + 1, len(self.cvn_config.regions)):
            mem_start, mem_end = self.cvn_config.regions[idx]
            file_start = mem_start - base
            file_end = mem_end - base
            if file_start >= 0 and file_end <= len(data):
                pos = file_start
                while pos + 3 <= file_end:
                    total_bytes_after += 4
                    pos += 4

        # A CRC difference propagates through one table step independently of the
        # data byte, since (crc1 ^ byte) ^ (crc2 ^ byte) = crc1 ^ crc2 = d
        def diff_step_table(d):
            return CRC32_TABLE[d & 0xFF] ^ (d >> 8)

        step_matrix = [diff_step_table(1 << i) for i in range(32)]

        # GF(2) matrix ops
        def matrix_mult_gf2(A, B):
            result = []
            for i in range(32):
                row = 0
                for j in range(32):
                    val = 0
                    for k in range(32):
                        if ((A[i] >> k) & 1) and ((B[k] >> j) & 1):
                            val ^= 1
                    row |= (val << j)
                result.append(row)
            return result

        def matrix_pow_gf2(M, n):
            if n == 0:
                return [1 << i for i in range(32)]
            result = [1 << i for i in range(32)]
            base = M[:]
            while n > 0:
                if n & 1:
                    result = matrix_mult_gf2(result, base)
                base = matrix_mult_gf2(base, base)
                n >>= 1
            return result

        def apply_matrix(matrix, vec):
            result = 0
            for out_bit in range(32):
                val = 0
                for in_bit in range(32):
                    if ((matrix[in_bit] >> out_bit) & 1) and ((vec >> in_bit) & 1):
                        val ^= 1
                result |= (val << out_bit)
            return result

        prop_matrix = matrix_pow_gf2(step_matrix, total_bytes_after)

        patch_effects = []
        for bit in range(32):
            final_effect = apply_matrix(prop_matrix, patch_effects_local[bit])
            patch_effects.append(final_effect)

        target_diff = target_cvn ^ cvn_with_zero

        # Augmented matrix, one row per output bit
        aug_matrix = []
        for out_bit in range(32):
            row = 0
            for in_bit in range(32):
                if (patch_effects[in_bit] >> out_bit) & 1:
                    row |= (1 << in_bit)
            target_bit = (target_diff >> out_bit) & 1
            aug_matrix.append([row, target_bit])

        # Gaussian elimination in GF(2)
        for col in range(32):
            pivot_row = None
            for row in range(col, 32):
                if (aug_matrix[row][0] >> col) & 1:
                    pivot_row = row
                    break

            if pivot_row is None:
                continue

            if pivot_row != col:
                aug_matrix[col], aug_matrix[pivot_row] = aug_matrix[pivot_row], aug_matrix[col]

            for row in range(32):
                if row != col and ((aug_matrix[row][0] >> col) & 1):
                    aug_matrix[row][0] ^= aug_matrix[col][0]
                    aug_matrix[row][1] ^= aug_matrix[col][1]

        # Back substitution
        patch_value = 0
        for row in range(32):
            row_matrix = aug_matrix[row][0]
            target_bit = aug_matrix[row][1]
            if row_matrix == 0:
                if target_bit != 0:
                    return False  # no solution
                continue
            for col in range(32):
                if (row_matrix >> col) & 1:
                    if target_bit:
                        patch_value |= (1 << col)
                    break

        # Write and verify
        struct.pack_into('<I', data, patch_offset, patch_value)
        new_cvn = self.calculate_cvn(bytes(data))

        return new_cvn == target_cvn

    def correct_add32_checksum(self, cs: ChecksumStructure, block_start_mem: int,
                                block_start_bin: int, data: bytearray) -> bool:
        """Correct an ADD32 checksum via the last dword of the checksummed region."""
        if cs.cs_algorithm != 0x01:
            return False

        start_offset = self._mem_to_file(cs.cs_start, block_start_mem, block_start_bin)
        end_offset = self._mem_to_file(cs.cs_end, block_start_mem, block_start_bin)

        if start_offset < 0 or end_offset > len(data) or start_offset >= end_offset:
            return False

        current_checksum = self.calculate_add32_checksum(start_offset, end_offset, cs.cs_start_val)
        target_checksum = cs.cs_expected_val

        difference = (target_checksum - current_checksum) & 0xFFFFFFFF

        # end_offset is inclusive, so the last dword starts 3 bytes back
        last_dword_offset = end_offset - 3
        old_value = struct.unpack('<I', data[last_dword_offset:last_dword_offset+4])[0]

        new_value = (old_value + difference) & 0xFFFFFFFF

        struct.pack_into('<I', data, last_dword_offset, new_value)

        return True

    def correct_add16_checksum(self, cs: ChecksumStructure, block_start_mem: int,
                                block_start_bin: int, data: bytearray) -> bool:
        """Correct an ADD16 checksum via the last dword of the checksummed region."""
        if cs.cs_algorithm != 0x10:
            return False

        start_offset = self._mem_to_file(cs.cs_start, block_start_mem, block_start_bin)
        end_offset = self._mem_to_file(cs.cs_end, block_start_mem, block_start_bin)

        if start_offset < 0 or end_offset > len(data) or start_offset >= end_offset:
            return False

        current_checksum = self.calculate_add16_checksum(start_offset, end_offset, cs.cs_start_val)
        target_checksum = cs.cs_expected_val

        difference = (target_checksum - current_checksum) & 0xFFFFFFFF

        # end_offset is inclusive, so the last dword starts 3 bytes back
        last_dword_offset = end_offset - 3
        old_value = struct.unpack('<I', data[last_dword_offset:last_dword_offset+4])[0]

        # The dword contributes low_word + high_word, so adding the difference to
        # the dword as a whole moves the sum by the same amount
        new_value = (old_value + difference) & 0xFFFFFFFF

        struct.pack_into('<I', data, last_dword_offset, new_value)

        return True

    def correct_crc32_checksum(self, cs: ChecksumStructure, block_start_mem: int,
                                block_start_bin: int, block_bin_end: int, data: bytearray) -> bool:
        """Correct a CRC32 checksum: forge the block signature, then solve dCSAdjust.

        RIPEMD-160 over the block up to the signature, forge a Bleichenbacher
        signature carrying that hash, write it, then solve the dCSAdjust dword
        that drives the region CRC32 to 0x35015001. ADD32/ADD16 must already be
        corrected, since this CRC covers those bytes.
        """
        if cs.cs_algorithm != 0x00:
            return False

        start_offset = self._mem_to_file(cs.cs_start, block_start_mem, block_start_bin)
        end_offset = self._mem_to_file(cs.cs_end, block_start_mem, block_start_bin)

        if start_offset < 0 or end_offset > len(data) or start_offset >= end_offset:
            return False

        # Epilog dCSAdjust is 4 bytes before DEADBEEF
        epilog_adjust_offset = block_bin_end - 7
        signature_offset = epilog_adjust_offset - 128

        if epilog_adjust_offset < start_offset or epilog_adjust_offset + 3 > end_offset:
            return False

        target_checksum = (~cs.cs_expected_val) & 0xFFFFFFFF

        # Hash covers the block up to the signature
        hash_start = block_start_bin
        hash_end = signature_offset
        block_data = bytes(data[hash_start:hash_end])

        ripemd160 = hashlib.new('ripemd160')
        ripemd160.update(block_data)
        ripemd_hash = ripemd160.digest()

        forged_signature = forge_bleichenbacher_signature(ripemd_hash)
        data[signature_offset:signature_offset+128] = forged_signature

        patch_value = solve_crc32_patch_matrix(
            bytes(data),
            start_offset,
            end_offset,
            epilog_adjust_offset,
            cs.cs_start_val,
            target_checksum
        )

        if patch_value is not None:
            # Write and verify
            struct.pack_into('<I', data, epilog_adjust_offset, patch_value)

            old_data = self.data
            self.data = bytes(data)
            verify_crc = self.calculate_crc32_algo(start_offset, end_offset, cs.cs_start_val)
            self.data = old_data

            return verify_crc == target_checksum
        else:
            return False

    def correct_all_checksums(self, output_path: Optional[str] = None) -> int:
        """Correct every invalid checksum, writing to output_path if given."""
        console.print()
        console.print(Panel("[bold cyan]Checksum Correction Process[/bold cyan]\n" +
                          "Two-pass algorithm: ADD32/ADD16 → CRC32",
                          border_style="cyan"))

        corrected_data = bytearray(self.data)
        corrected_count = 0

        # Reserved partitions aren't real firmware — correcting them would
        # fabricate a checksum for empty space and mutate an original image
        for block in self.bosch_blocks:
            if block.is_unprogrammed:
                print_info(f"{block.block_name}: unprogrammed/reserved "
                           f"(checksum fields are 0x{block.checksum:08X} fill) — left unchanged")

        # Pass 1: ADD32/ADD16
        console.print()
        console.print("[bold blue]PASS 1:[/bold blue] Correcting ADD32 and ADD16 checksums")
        console.print()

        for i, block in enumerate(self.bosch_blocks, 1):
            if block.is_unprogrammed:
                continue
            has_add = any(cs.cs_algorithm in (0x01, 0x10) for cs in block.checksum_structures)
            if not has_add:
                continue

            console.print(f"[yellow]Block {i}:[/yellow] {block.block_name}")

            for j, cs in enumerate(block.checksum_structures, 1):
                if cs.cs_algorithm not in (0x01, 0x10):
                    continue

                self.data = bytes(corrected_data)
                self.validate_checksum_structure(cs, block.block_start, block.bin_start)

                algo_name = "ADD32" if cs.cs_algorithm == 0x01 else "ADD16"

                if cs.is_valid:
                    console.print(f"  Structure {j} ({algo_name}): [green]✓[/green] Already valid")
                    continue

                console.print(f"  Structure {j} ({algo_name}): [red]✗[/red] Invalid (0x{cs.calculated_checksum:08X})")

                if cs.cs_algorithm == 0x01:
                    success = self.correct_add32_checksum(cs, block.block_start,
                                                          block.bin_start, corrected_data)
                else:  # 0x10
                    success = self.correct_add16_checksum(cs, block.block_start,
                                                          block.bin_start, corrected_data)

                if success:
                    corrected_count += 1
                    self.data = bytes(corrected_data)
                    self.validate_checksum_structure(cs, block.block_start, block.bin_start)
                    if cs.is_valid:
                        print_success(f"Corrected to 0x{cs.calculated_checksum:08X}")
                    else:
                        print_error("Verification failed")
                else:
                    print_error("Correction failed")

        # Per-variant monitoring checksums, before CRC32 for the same reason
        if self.find_variant_config(bytes(corrected_data)):
            console.print()
            console.print("[bold blue]PASS 1b:[/bold blue] Correcting per-variant "
                          "monitoring checksums")
            console.print()
            corrected_count += self.correct_variant_checksums(corrected_data)
            self.data = bytes(corrected_data)

        # Pass 2: CRC32, which must follow pass 1 — it covers the ADD-corrected bytes
        console.print()
        console.print("[bold blue]PASS 2:[/bold blue] Correcting CRC32 checksums")
        console.print()

        for i, block in enumerate(self.bosch_blocks, 1):
            if block.is_unprogrammed:
                continue
            has_crc32 = any(cs.cs_algorithm == 0x00 for cs in block.checksum_structures)
            if not has_crc32:
                continue

            console.print(f"[yellow]Block {i}:[/yellow] {block.block_name}")

            for j, cs in enumerate(block.checksum_structures, 1):
                if cs.cs_algorithm != 0x00:
                    continue

                self.data = bytes(corrected_data)
                self.validate_checksum_structure(cs, block.block_start, block.bin_start)

                if cs.is_valid:
                    console.print(f"  Structure {j}: [green]✓[/green] Already valid")
                    continue

                console.print(f"  Structure {j}: [red]✗[/red] Invalid (0x{cs.calculated_checksum:08X})")

                success = self.correct_crc32_checksum(cs, block.block_start,
                                                      block.bin_start, block.bin_end,
                                                      corrected_data)

                if success:
                    corrected_count += 1
                    self.data = bytes(corrected_data)
                    self.validate_checksum_structure(cs, block.block_start, block.bin_start)
                    if cs.is_valid:
                        print_success(f"Corrected to 0x{cs.calculated_checksum:08X}")
                    else:
                        print_error("Verification failed")
                else:
                    print_error("Correction failed")

        self.data = bytes(self.data)

        console.print()
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(corrected_data)

            if corrected_count > 0:
                console.print(Panel(
                    f"[green]✓[/green] Corrected binary saved to:\n[cyan]{output_path}[/cyan]\n\n" +
                    f"[bold]Checksums corrected:[/bold] {corrected_count}",
                    title="💾 Success",
                    border_style="green"
                ))
            else:
                print_info("All checksums already valid - no corrections needed")
        elif corrected_count > 0:
            print_warning("Corrections made but not saved (specify output path)")
        else:
            print_info("All checksums already valid - no corrections needed")

        return corrected_count

    def checksum_to_dict(self, cs: ChecksumStructure, block: BoschBlock, index: int) -> dict:
        error = None
        if cs.cs_algorithm not in self.ALGORITHM_NAMES:
            error = f"Unsupported algorithm 0x{cs.cs_algorithm:02X}"
        elif cs.calculated_checksum is None:
            error = 'Checksum region lies outside the binary'

        # CRC32 stores the complement of the value it compares against
        expected = ((~cs.cs_expected_val) & 0xFFFFFFFF
                    if cs.cs_algorithm == 0x00
                    else cs.cs_expected_val)

        return {
            'index': index,
            'algorithm': self.ALGORITHM_NAMES.get(cs.cs_algorithm, 'UNKNOWN'),
            'offset': hex32(cs.offset),
            'start': hex32(cs.cs_start),
            'end': hex32(cs.cs_end),
            'length': max(cs.cs_end - cs.cs_start + 1, 0),
            'valid': None if block.is_unprogrammed else cs.is_valid,
            'expected': None if block.is_unprogrammed else hex32(expected),
            'calculated': hex32(cs.calculated_checksum),
            'error': error,
        }

    def block_to_dict(self, block: BoschBlock, index: int) -> dict:
        return {
            'index': index,
            'name': block.block_name,
            'identifier': hex32(block.block_identifier),
            'type_id': f"0x{block.block_type_id:02X}",
            'file_start': hex32(block.bin_start),
            'file_end': hex32(block.bin_end),
            'memory_start': hex32(block.block_start),
            'memory_end': hex32(block.block_end),
            'size': block.size,
            'sw_identifier': block.sw_identifier.decode('ascii', errors='ignore').strip('\x00 '),
            'has_otp': block.has_otp,
            'is_unprogrammed': block.is_unprogrammed,
            'checksums': [
                self.checksum_to_dict(cs, block, i)
                for i, cs in enumerate(block.checksum_structures, 1)
            ],
        }

    def to_dict(self) -> dict:
        """Full validation state as plain data, for --json."""
        variants = self.identify_ecu_variant()

        document = {
            'filename': self.binary_path.name,
            'file_size': len(self.data),
            'platform': 'MED17/EDC17',
            'ecu_variant': variants[0] if variants else None,
            'total_blocks': len(self.bosch_blocks),
        }
        document.update(self.checksum_counts())
        document['blocks'] = [
            self.block_to_dict(block, i)
            for i, block in enumerate(self.bosch_blocks, 1)
        ]
        document['cvn'] = self.cvn_feasibility()
        document['variant_dataset'] = self.variant_to_dict()

        return document

    def print_summary(self) -> None:
        console.print()

        file_info = f"[cyan]{self.binary_path.name}[/cyan]\n"
        file_info += f"Size: 0x{len(self.data):X} ({len(self.data):,} bytes)"
        console.print(Panel(file_info, title="📁 Binary File", border_style="cyan"))

        variants = self.identify_ecu_variant()
        if variants:
            variant_text = "\n".join(f"• {v}" for v in variants)
            console.print(Panel(variant_text, title="ECU Variant", border_style="blue"))

        if self.cvn_config and self.cvn_config.calculated_cvn is not None:
            cvn_text = f"[bold]CVN:[/bold] 0x{self.cvn_config.calculated_cvn:08X}\n"
            cvn_text += f"[dim]Regions: {len(self.cvn_config.regions)}[/dim]"
            console.print(Panel(cvn_text, title="CVN (Calibration Verification Number)", border_style="magenta"))

        console.print()
        console.print(f"[bold cyan]═══ Bosch Checksum Blocks ({len(self.bosch_blocks)} found) ═══[/bold cyan]")

        for i, block in enumerate(self.bosch_blocks, 1):
            console.print()

            otp_indicator = " [red][OTP][/red]" if block.has_otp else ""
            reserved_indicator = " [yellow][RESERVED][/yellow]" if block.is_unprogrammed else ""
            header = (f"[bold yellow]Block {i}:[/bold yellow] [cyan]{block.block_name}[/cyan]"
                      f"{otp_indicator}{reserved_indicator}")
            console.print(header)

            info_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
            info_table.add_column("Property", style="dim")
            info_table.add_column("Value")

            info_table.add_row("Location", f"0x{block.bin_start:08X} - 0x{block.bin_end:08X}")
            info_table.add_row("Memory", f"0x{block.block_start:08X} - 0x{block.block_end:08X}")
            info_table.add_row("Size", f"0x{block.size:X} ({block.size:,} bytes)")
            info_table.add_row("Identifier", f"0x{block.block_identifier:08X} (type: 0x{block.block_type_id:02X})")

            console.print(info_table)

            if block.checksum_structures:
                console.print()
                cs_table = Table(title=f"Checksum Structures ({len(block.checksum_structures)})",
                               box=box.ROUNDED, show_lines=True)

                cs_table.add_column("#", style="dim", width=3)
                cs_table.add_column("Algorithm", width=10)
                cs_table.add_column("Range", width=25)
                cs_table.add_column("Calculated", width=10, justify="right")
                cs_table.add_column("Expected", width=10, justify="right")
                cs_table.add_column("Status", width=10, justify="center")

                for j, cs in enumerate(block.checksum_structures, 1):
                    algo_name = self.ALGORITHM_NAMES.get(cs.cs_algorithm, "UNKNOWN")
                    range_str = f"0x{cs.cs_start:08X}\n0x{cs.cs_end:08X}"

                    if block.is_unprogrammed:
                        # No finalised checksum to compare against — don't flag as failed
                        calc_str = f"0x{cs.calculated_checksum:08X}" if cs.calculated_checksum is not None else "-"
                        exp_str = "-"
                        status = Text("⊘ RESERVED", style="yellow")
                    elif cs.calculated_checksum is not None:
                        calc_str = f"0x{cs.calculated_checksum:08X}"
                        exp_str = (f"0x{(~cs.cs_expected_val) & 0xFFFFFFFF:08X}"
                                   if cs.cs_algorithm == 0x00
                                   else f"0x{cs.cs_expected_val:08X}")

                        if cs.is_valid:
                            status = Text("✓ VALID", style="bold green")
                        else:
                            status = Text("✗ INVALID", style="bold red")
                    else:
                        calc_str = "-"
                        exp_str = "-"
                        status = Text("?", style="dim")

                    cs_table.add_row(str(j), algo_name, range_str, calc_str, exp_str, status)

                console.print(cs_table)

        console.print()
        console.print("[dim]" + "═" * 70 + "[/dim]")

    def parse(self) -> None:
        """Main parsing routine"""
        self.load_binary()
        self.find_bosch_blocks()
        self.validate_all_checksums()

        self.cvn_config = self.find_cvn_config()
        if self.cvn_config:
            self.cvn_config.calculated_cvn = self.calculate_cvn()
            console.print(f"[+] CVN: 0x{self.cvn_config.calculated_cvn:08X}")

        cfg = self.find_variant_config()
        if cfg:
            console.print(f"[+] Variant dataset: {cfg.count} variants, "
                          f"{len(cfg.tables)} tables found")
            console.print(f"    Monitored region 0x{cfg.region_start:08X}-"
                          f"0x{cfg.region_end:08X}, compensation at "
                          f"0x{cfg.adjust_offset:08X}")
            if len(cfg.tables) + 1 != cfg.count:
                print_warning(f"Header declares {cfg.count} variants but "
                              f"{len(cfg.tables) + 1} tables were located")

        if not QUIET:
            self.print_summary()


class ToolError(Exception):
    """A fatal, user-facing failure: reported as JSON or plain text, exit code 2."""


def fail(message: str, as_json: bool, **extra) -> None:
    """Report a fatal error in the caller's chosen format and exit 2."""
    if as_json:
        document = {'error': message}
        document.update(extra)
        json.dump(document, sys.stdout)
        print()
    else:
        print_error(message)

    sys.exit(2)


def run_correction(parser: 'MEDC17BinaryParser', args, output_path: str) -> dict:
    """Preserve the CVN if asked, correct the checksums, and report the outcome.

    CVN first: its compensation dword lands inside the checksummed range, so the
    checksum passes have to run over the final bytes.
    """
    corrected_data = bytearray(parser.data)
    errors: List[str] = []
    warnings: List[str] = []

    report = {
        'success': False,
        'checksums_corrected': 0,
        'output_path': output_path,
        'cvn_mode': 'original' if args.fix_cvn else ('best_effort' if args.fix_cvn_inplace else 'none'),
        'cvn_applied': False,
        'cvn_before': parser.cvn_feasibility()['value'],
        'cvn_after': None,
        'target_cvn': None,
        'errors': errors,
        'warnings': warnings,
    }

    target_cvn = None

    if args.fix_cvn:
        console.print()
        console.print(Panel("[bold cyan]CVN Correction[/bold cyan]", border_style="cyan"))

        if not Path(args.fix_cvn).exists():
            raise ToolError(f"Original file not found: {args.fix_cvn}")

        original_parser = MEDC17BinaryParser(args.fix_cvn)
        original_parser.load_binary()
        original_parser.find_bosch_blocks()
        original_parser.cvn_config = original_parser.find_cvn_config()

        if original_parser.cvn_config is None:
            raise ToolError("Could not find CVN config in original file")

        target_cvn = original_parser.calculate_cvn()
        report['target_cvn'] = hex32(target_cvn)
        print_info(f"Target CVN (from original): 0x{target_cvn:08X}")

        parser.data = bytes(corrected_data)
        current_cvn = parser.calculate_cvn()
        print_info(f"Current CVN: 0x{current_cvn:08X}")

        if current_cvn == target_cvn:
            print_success("CVN already matches target")
        else:
            console.print("[yellow]Correcting CVN...[/yellow]")
            if not parser.correct_cvn(target_cvn, corrected_data):
                raise ToolError("CVN correction failed")
            print_success(f"CVN patched for target 0x{target_cvn:08X}")

        report['cvn_applied'] = True

    if args.fix_cvn_inplace:
        console.print()
        console.print(Panel("[bold cyan]CVN Correction (best effort, no original)[/bold cyan]",
                            border_style="cyan"))

        parser.data = bytes(corrected_data)

        if parser.cvn_config is None:
            raise ToolError("Could not find CVN config in input file")

        status = parser.cvn_feasibility(corrected_data)
        if status['preserve_reason'] and not status['preserve_supported']:
            raise ToolError(f"CVN cannot be preserved: {status['preserve_reason']}")

        print_info(f"Current CVN: 0x{parser.calculate_cvn():08X}")

        if not parser.correct_cvn_best_effort(corrected_data):
            raise ToolError("Best-effort CVN correction failed")

        parser.data = bytes(corrected_data)
        print_info(f"Preserved CVN: 0x{parser.calculate_cvn():08X}")
        report['cvn_applied'] = True

    # Checksums last — the CVN patch lands inside their range
    parser.data = bytes(corrected_data)
    report['checksums_corrected'] = parser.correct_all_checksums(output_path)

    # Re-validate what actually landed on disk, rather than trusting the state
    # the correction passes left behind
    with open(output_path, 'rb') as f:
        final_data = f.read()

    parser.data = final_data
    parser.validate_all_checksums()
    counts = parser.checksum_counts()

    report['after'] = {
        'all_valid': counts['all_valid'],
        'total_checksums': counts['total_checksums'],
        'valid_checksums': counts['valid_checksums'],
        'invalid_checksums': counts['invalid_checksums'],
        'cvn': None,
        'comptest_matches': None,
    }

    if counts['invalid_checksums']:
        errors.append(f"{counts['invalid_checksums']} checksum(s) still invalid after correction")

    if counts['unknown_checksums']:
        warnings.append(f"{counts['unknown_checksums']} checksum(s) use an unsupported algorithm "
                        "and were left unchanged")

    if parser.cvn_config is not None:
        final_status = parser.cvn_feasibility(final_data)
        final_cvn = parser.calculate_cvn()
        report['cvn_after'] = final_status['value']
        report['after']['cvn'] = final_status['value']

        if args.fix_cvn:
            console.print()
            console.print("[dim]Verifying CVN after checksum correction...[/dim]")
            if final_cvn == target_cvn:
                print_success(f"CVN verified: 0x{final_cvn:08X}")
            else:
                print_error(f"CVN verification failed: got 0x{final_cvn:08X}")
                errors.append(f"CVN verification failed: got {hex32(final_cvn)}, "
                              f"expected {hex32(target_cvn)}")

        if args.fix_cvn_inplace:
            console.print()
            console.print("[dim]Verifying CVN after checksum correction...[/dim]")
            report['after']['comptest_matches'] = final_status['comptest_matches']

            if final_status['comptest_matches']:
                print_success(f"CompTest CRC verified: {final_status['current_comptest_crc']} "
                              f"(CVN preserved: {hex32(final_cvn)})")
            else:
                print_error(f"CompTest verification failed: got {final_status['current_comptest_crc']}, "
                            f"expected {final_status['stored_comptest_crc']}")
                errors.append('CompTest CRC no longer matches — the CVN was not preserved')

    report['success'] = not errors and counts['all_valid']

    return report


def parse_arguments():
    import argparse

    parser_args = argparse.ArgumentParser(
        description='MEDC17 Checksum Analyzer & Corrector v1.1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Analyze binary and validate checksums
  %(prog)s firmware.bin

  # Correct invalid checksums and save to new file
  %(prog)s firmware.bin --correct -o firmware_fixed.bin

  # Correct checksums and overwrite original (use with caution!)
  %(prog)s firmware.bin --correct --overwrite

  # Correct checksums AND CVN to match original file
  %(prog)s modified.bin --correct --fix-cvn original.bin -o fixed.bin

  # Correct checksums AND preserve CVN without the original file
  %(prog)s modified.bin --correct --fix-cvn-inplace -o fixed.bin

  # Machine-readable report, no third-party dependencies
  %(prog)s firmware.bin --json

Exit codes:
  0  all checksums valid (or every correction succeeded)
  1  invalid checksums found (or a correction did not succeed)
  2  error — file missing, not a MED17/EDC17 binary, bad arguments
        '''
    )

    parser_args.add_argument('binary_file', help='Input binary file to analyze')
    parser_args.add_argument('--correct', '-c', action='store_true',
                           help='Correct invalid checksums')
    parser_args.add_argument('--output', '-o', metavar='FILE',
                           help='Output file for corrected binary')
    parser_args.add_argument('--overwrite', action='store_true',
                           help='Overwrite input file with corrections (dangerous!)')
    parser_args.add_argument('--fix-cvn', metavar='ORIGINAL',
                           help='Fix CVN to match the CVN from ORIGINAL file')
    parser_args.add_argument('--fix-cvn-inplace', action='store_true',
                           help='Preserve CVN without the original file (best effort, '
                                'via the stored CompTest CRC)')
    parser_args.add_argument('--json', action='store_true',
                           help='Emit one JSON document on stdout and nothing else '
                                '(for programmatic use)')

    return parser_args.parse_args()


def main():
    """Main entry point"""
    import time

    args = parse_arguments()
    set_quiet(args.json)

    if not args.json:
        print_banner()

    try:
        if args.correct and args.overwrite and args.output:
            raise ToolError('Cannot specify both --output and --overwrite')

        if args.fix_cvn and args.fix_cvn_inplace:
            raise ToolError('Cannot specify both --fix-cvn and --fix-cvn-inplace')

        start_time = time.time()

        parser = MEDC17BinaryParser(args.binary_file)

        try:
            parser.parse()
        except FileNotFoundError:
            raise ToolError(f"File not found: {args.binary_file}")
        except OSError as e:
            raise ToolError(f"Could not read {args.binary_file}: {e}")

        if not parser.bosch_blocks:
            raise ToolError('No Bosch checksum blocks found — is this a MED17/EDC17 binary?')

        # Snapshot the incoming state before anything is corrected
        document = parser.to_dict() if args.json else None

        output_path = args.output or (args.binary_file if args.overwrite else None)
        wants_correction = args.correct or args.fix_cvn or args.fix_cvn_inplace

        correction = None
        if wants_correction and output_path is None:
            if args.json:
                raise ToolError('Correction requested but no output path given')

            console.print()
            print_warning('Correction requested but no output path given')
            print_info('Use --output <file> or --overwrite to save corrections')
        elif wants_correction:
            correction = run_correction(parser, args, output_path)

        if args.json:
            document['correction'] = correction
            json.dump(document, sys.stdout)
            print()
        else:
            console.print()
            console.print(f"[dim]Completed in {time.time() - start_time:.2f}s[/dim]")

        if correction is not None:
            sys.exit(0 if correction['success'] else 1)

        sys.exit(0 if parser.checksum_counts()['all_valid'] else 1)

    except ToolError as e:
        fail(str(e), args.json, filename=os.path.basename(args.binary_file))
    except Exception as e:
        if not args.json:
            import traceback
            traceback.print_exc()

        fail(f"Unexpected error: {e}", args.json, filename=os.path.basename(args.binary_file))


if __name__ == "__main__":
    main()
