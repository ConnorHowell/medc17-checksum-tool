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

import struct
import sys
import hashlib
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.text import Text

# Rich console for styled output
console = Console()


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
    """Print success message in green."""
    console.print(f"✓ {message}", style="bold green")


def print_error(message: str):
    """Print error message in red."""
    console.print(f"✗ {message}", style="bold red")


def print_info(message: str):
    """Print info message in blue."""
    console.print(f"ℹ {message}", style="blue")


def print_warning(message: str):
    """Print warning message in yellow."""
    console.print(f"⚠ {message}", style="yellow")


# Pre-computed CRC32 lookup table (IEEE 802.3 polynomial: 0xEDB88320)
# Generated once for performance optimization
CRC32_TABLE = None

def init_crc32_table():
    """Initialize CRC32 lookup table for fast calculation"""
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


def reverse_bits_32(n):
    """Reverse all 32 bits of an integer"""
    result = 0
    for i in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


def reverse_bits_8(n):
    """Reverse 8 bits of a byte"""
    result = 0
    for i in range(8):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


def crc32_forge_patch(current_crc, target_crc):
    """
    Calculate 4-byte patch value to transform current_crc to target_crc.

    This assumes the patch bytes are at the end of the checksummed region.
    Based on the inverse CRC algorithm.

    Args:
        current_crc: Current CRC32 value (with patch bytes as zeros)
        target_crc: Desired CRC32 value

    Returns:
        4-byte value (as int) to use as patch
    """
    # Step 1: XOR current and target to get difference
    diff = current_crc ^ target_crc

    # Step 2: Bit-reverse the 32-bit difference
    diff_reversed = reverse_bits_32(diff)

    # Step 3: Multiply by modular inverse of CRC polynomial in GF(2)
    # For CRC32 (polynomial 0x104C11DB7), the inverse is 0xDB710641
    # This is pre-calculated using extended Euclidean algorithm in GF(2)
    poly_inverse = 0xDB710641

    # Perform GF(2) multiplication (without modulo since we're working with inverse)
    result = gf2_multiply(diff_reversed, poly_inverse)

    # Step 4: Byte-wise bit reversal
    patch_bytes = []
    for i in range(4):
        byte_val = (result >> (i * 8)) & 0xFF
        patch_bytes.append(reverse_bits_8(byte_val))

    # Pack as little-endian 32-bit value
    patch_value = struct.unpack('<I', bytes(patch_bytes))[0]

    return patch_value


def gf2_multiply(a, b):
    """
    Multiply two values in GF(2) field (XOR-based multiplication).
    Used for CRC forging calculations.
    """
    result = 0
    for i in range(32):
        if b & 1:
            result ^= a
        a <<= 1
        if a & 0x100000000:
            a ^= 0x104C11DB7  # CRC32 polynomial
        b >>= 1
    return result & 0xFFFFFFFF


# Block identifiers from old_parser (observed in binaries)
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


def cube_root_int(n):
    """Calculate integer cube root using Newton's method."""
    if n == 0:
        return 0
    x = n
    y = (2 * x + n // (x * x)) // 3
    while y < x:
        x = y
        y = (2 * x + n // (x * x)) // 3
    return x


def forge_bleichenbacher_signature(ripemd_hash: bytes) -> bytes:
    """
    Forge a Bleichenbacher signature for RIPEMD-160 with e=3.

    Uses a SIMPLIFIED format (not full PKCS#1 v1.5):
    - RSA with e=3, NO modulus (just cubes the signature)
    - When cubed, should be: 01 FF FF ... FF 00 [20-byte hash] [garbage]
    - Note: starts with 01 (not 00 01), hash is raw without DigestInfo wrapper

    Args:
        ripemd_hash: 20-byte RIPEMD-160 hash

    Returns:
        128-byte forged signature
    """
    # Build target for signature^3:
    # Format: 01 FF*8 00 [hash] [padding]
    # Total: 127 bytes (to avoid leading 00 in result)

    target = bytearray(127)
    target[0] = 0x01
    for i in range(1, 9):  # 8 bytes of FF padding
        target[i] = 0xFF
    target[9] = 0x00
    target[10:30] = ripemd_hash  # 20-byte hash
    # Rest is zeros/garbage

    # Convert to integer and find cube root
    target_int = int.from_bytes(bytes(target), 'big')
    sig_int = cube_root_int(target_int)

    # Try values around cube root to find best match
    best_sig = sig_int
    for candidate in [sig_int - 1, sig_int, sig_int + 1, sig_int + 2]:
        if candidate < 0:
            continue
        cubed = candidate ** 3
        # We want the largest value where cubed <= target
        if cubed <= target_int and candidate > best_sig:
            best_sig = candidate

    # Pad to 128 bytes
    sig_bytes = best_sig.to_bytes(128, 'big')
    return sig_bytes


def crc32_process_dword_bitwise(initial_crc: int, dword_input: int) -> int:
    """Process a single dword through CRC32 bit-by-bit algorithm."""
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
    """
    Build 32x32 transformation matrix for CRC32 operation in GF(2).

    Shows how each bit of input dword affects output CRC.
    Returns list of 32 integers (rows as bitmasks).
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
    """
    Solve linear system in GF(2): M * x = target
    Uses Gaussian elimination with XOR arithmetic.
    """
    # Create augmented matrix [M | target]
    aug_matrix = []
    for i in range(32):
        target_bit = (target >> i) & 1
        aug_matrix.append([matrix[i], target_bit])

    # Forward elimination
    for col in range(32):
        # Find pivot
        pivot_row = None
        for row in range(col, 32):
            if (aug_matrix[row][0] >> col) & 1:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        # Swap pivot to current position
        if pivot_row != col:
            aug_matrix[col], aug_matrix[pivot_row] = aug_matrix[pivot_row], aug_matrix[col]

        # Eliminate column in other rows
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

        # Find leading 1
        for col in range(32):
            if (row_matrix >> col) & 1:
                if target_bit:
                    solution |= (1 << col)
                break

    return solution


def solve_crc32_patch_matrix(data: bytes, start_offset: int, end_offset: int,
                             patch_offset: int, initial_value: int, target_crc: int) -> Optional[int]:
    """
    Solve for CRC32 patch value using matrix algebra in GF(2).

    Mathematical solution that works instantly instead of iterative search.
    Exploits the linearity of CRC over GF(2).
    """
    if patch_offset < start_offset or patch_offset + 3 > end_offset:
        return None

    # Calculate intermediate CRC up to (but not including) patch
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

    # Build transformation matrix
    matrix = build_crc_transformation_matrix(intermediate_crc)

    # Calculate CRC with patch=0
    data_copy = bytearray(data)
    struct.pack_into('<I', data_copy, patch_offset, 0x00000000)
    crc_with_zero = calc_crc_range(bytes(data_copy), start_offset, end_offset, initial_value)

    # Solve: matrix * patch = (target XOR crc_with_zero)
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
    cs_algorithm: int  # Algorithm identifier (0=?, 1=?, 0x10=CRC32?)
    calculated_checksum: Optional[int] = None
    is_valid: Optional[bool] = None


@dataclass
class CVNConfig:
    """CVN (Calibration Verification Number) configuration"""
    config_offset: int  # File offset of the config structure
    regions: List[tuple]  # List of (start, end) memory addresses
    ds_start: int  # Dataset start memory address
    ds_wocs_end: int  # Dataset WOCS (without checksum) end memory address
    base_address: int  # Memory base address (for file offset conversion)
    calculated_cvn: Optional[int] = None


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
        """Get the block type from the first byte of the identifier"""
        return self.block_identifier & 0xFF

    @property
    def has_otp(self) -> bool:
        """Check if this block has OTP (One-Time Programmable) flag set"""
        return bool(self.block_identifier & 0x00800000)


class MEDC17BinaryParser:
    """Parser for MEDC17 ECU binary files (little-endian format)"""

    CHECKSUM_STRUCTURE_SIZE = 32  # Each checksum structure is 32 bytes

    def __init__(self, binary_path: str):
        """Initialize parser with binary file path"""
        self.binary_path = Path(binary_path)
        self.data: bytes = b''
        self.bosch_blocks: List[BoschBlock] = []
        self.cvn_config: Optional[CVNConfig] = None

    def load_binary(self) -> None:
        """Load binary file into memory"""
        if not self.binary_path.exists():
            raise FileNotFoundError(f"Binary file not found: {self.binary_path}")

        with open(self.binary_path, 'rb') as f:
            self.data = f.read()

        print_success(f"Loaded: {self.binary_path.name}")
        print_info(f"Size: 0x{len(self.data):X} ({len(self.data):,} bytes)")

    def read_dword_le(self, offset: int) -> int:
        """Read 32-bit little-endian value"""
        if offset + 4 > len(self.data):
            return 0
        return struct.unpack('<I', self.data[offset:offset+4])[0]

    def read_word_le(self, offset: int) -> int:
        """Read 16-bit little-endian value"""
        if offset + 2 > len(self.data):
            return 0
        return struct.unpack('<H', self.data[offset:offset+2])[0]

    def read_byte(self, offset: int) -> int:
        """Read single byte"""
        if offset >= len(self.data):
            return 0
        return self.data[offset]

    def find_next_nonzero(self, start: int) -> Optional[int]:
        """Find next non-zero byte starting from offset"""
        for i in range(start, len(self.data)):
            if self.data[i] != 0:
                return i
        return None

    def read_checksum_structures(self, offset: int, count: int) -> List[ChecksumStructure]:
        """Read checksum structures (32 bytes each)"""
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
                cs_algorithm=self.read_word_le(struct_offset + 28) & 0xFF,  # Read lower byte of algorithm ID
            ))

        return structures

    def parse_block(self, flat_address: int) -> Optional[BoschBlock]:
        """
        Parse Bosch block at given offset

        Block structure (little-endian):
        +0x00: Block identifier (dword)
        +0x04: Size (dword)
        +0x0C: Block end address (dword)
        +0x1A: Software identifier (10 bytes)
        +0x2C: Number of checksum structures (dword)
        +0x30: Checksum adjust (dword)
        +0x34: Checksum structures start (32 bytes each)
        Last: Final checksum (dword)

        Returns None if the data at flat_address doesn't look like a valid block.
        """
        if flat_address + 0x40 > len(self.data):
            return None

        # Read block header
        block_identifier = self.read_dword_le(flat_address)
        size = self.read_dword_le(flat_address + 4)
        block_end = self.read_dword_le(flat_address + 12)

        # Validate block type ID is known
        block_type_id = block_identifier & 0xFF
        if block_type_id not in BLOCK_IDENTIFIERS:
            return None

        # Validate size is reasonable (must fit in file and be > minimum header size)
        if size < 0x40 or size > len(self.data) or flat_address + size > len(self.data):
            return None

        # Validate block ends with DEADBEEF marker (little-endian)
        block_end_offset = flat_address + size - 4
        if self.read_dword_le(block_end_offset) != 0xDEADBEEF:
            return None

        # Validate block_end is a valid TriCore flash address
        if not self._is_flash_addr(block_end):
            return None

        # Software identifier at offset +0x1A (26)
        identifier_length = 10
        sw_identifier = self.data[flat_address + 26:flat_address + 26 + identifier_length]

        # Number of checksum structures at offset +0x2C (26 + 10 + 8)
        num_checksum_structures = self.read_dword_le(flat_address + 26 + identifier_length + 8)

        # Validate number of checksum structures is reasonable
        if num_checksum_structures > 100:
            return None

        # Calculate block start from block_end and size
        block_start = ((block_end + 5) - size - 1)

        # Validate block_start is a valid TriCore flash address
        if not self._is_flash_addr(block_start):
            return None

        # Checksum adjust at offset +0x30 (48)
        checksum_adjust = self.read_dword_le(flat_address + 0x30)

        # Read checksum structures starting at +0x34 (52)
        checksum_structures = self.read_checksum_structures(
            flat_address + 0x34,
            num_checksum_structures
        )

        # Final checksum after all checksum structures
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

    def find_bosch_blocks(self) -> None:
        """Find all Bosch blocks by scanning for non-zero bytes after padding"""
        print("\n[*] Scanning for Bosch checksum blocks...")

        # Clear existing blocks before re-scanning
        self.bosch_blocks = []

        # Find first block (first non-zero byte)
        current_pos = self.find_next_nonzero(0)
        if current_pos is None:
            print("[!] No blocks found (file is all zeros)")
            return

        block_count = 0

        while current_pos is not None and current_pos < len(self.data):
            block = self.parse_block(current_pos)

            if block is None:
                # Not a valid block - silently skip to next non-zero region
                next_pos = self.find_next_nonzero(current_pos + 1)
                if next_pos is None or next_pos >= len(self.data):
                    break
                current_pos = next_pos
                continue

            print(f"[+] Found block {block_count + 1} at 0x{current_pos:X}: {block.block_name}")
            self.bosch_blocks.append(block)
            block_count += 1

            # Find next block after this one
            next_pos = self.find_next_nonzero(block.bin_end + 1)
            if next_pos is None or next_pos >= len(self.data):
                break

            current_pos = next_pos

        print(f"[+] Total Bosch blocks found: {len(self.bosch_blocks)}")

    def identify_ecu_variant(self) -> List[str]:
        """
        Identify ECU variant by reading variant string from Dataset block. (This could probably be done a smarter way)

        The Dataset #0 block (ID 0x60) contains variant information at offset 0x78.
        Format: slash-separated fields, one contains the ECU variant (e.g., "EDC17_C46")
        Example: "34/1/EDC17_C46/5/P643//C643X5L8///"
        """
        # Look for Dataset #0 block (ID 0x60)
        dataset_block = None
        for block in self.bosch_blocks:
            if block.block_identifier == 0x60:
                dataset_block = block
                break

        if not dataset_block:
            return ["Unknown (no Dataset block found)"]

        # Read variant string at offset 0x78 from block start
        variant_offset = dataset_block.bin_start + 0x78

        if variant_offset + 100 > len(self.data):  # Safety check
            return ["Unknown (offset out of range)"]

        # Read up to 100 bytes or until null terminator
        variant_data = self.data[variant_offset:variant_offset+100]

        # Find null terminator or end
        null_pos = variant_data.find(b'\x00')
        if null_pos != -1:
            variant_data = variant_data[:null_pos]

        try:
            variant_string = variant_data.decode('ascii', errors='ignore')
        except:
            return ["Unknown (decode error)"]

        # Split by slashes and find the field containing MED17 or EDC17
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
            # Fallback: show all non-empty fields
            non_empty = [f for f in fields if f.strip()]
            if non_empty:
                return [f"Unknown variant (fields: {', '.join(non_empty[:3])})"]
            else:
                return ["Unknown"]

    def calculate_crc32_algo(self, start: int, end_inclusive: int, initial_value: int) -> int:
        """
        Calculate CRC32 checksum (algorithm 0x00 = SB_CRC32_ALGO_E).
        Bit-by-bit CRC32-IEEE with little-endian dwords (TriCore MCU format).

        Algorithm implementation:
        - Polynomial: 0xEDB88320
        - Reads dwords as little-endian (TriCore MCU native format)
        - Processes each bit of each dword
        - Expected result: 0x35015001 (complement of 0xCAFEAFFE)

        Note: This uses bit-by-bit processing for accuracy.
        A lookup table version could be faster but might not match exactly.

        Args:
            start: Start offset in binary
            end_inclusive: End offset in binary (inclusive - last byte to checksum)
            initial_value: Initial CRC value (usually 0xFADECAFE)

        Returns:
            32-bit CRC value
        """
        if start < 0 or end_inclusive >= len(self.data) or start > end_inclusive:
            return 0

        crc = initial_value
        pos = start

        # Process data as little-endian dwords up to and including end_inclusive
        while pos + 3 <= end_inclusive:
            # Read as little-endian dword (TriCore native format)
            dword = struct.unpack('<I', self.data[pos:pos+4])[0]
            pos += 4

            # Process each bit of the dword (32 bits)
            for _ in range(32):
                xor_result = dword ^ crc
                dword >>= 1
                if (xor_result & 1) != 0:
                    crc = (crc >> 1) ^ 0xEDB88320
                else:
                    crc = crc >> 1

        return crc

    def calculate_add32_checksum(self, start: int, end_inclusive: int, initial_value: int) -> int:
        """
        Calculate ADD32 checksum (algorithm 0x01 = SB_ADD32_ALGO_E).
        Simple 32-bit addition of all dwords.

        Args:
            start: Start offset in binary
            end_inclusive: End offset in binary (inclusive)
            initial_value: Initial checksum value (usually 0xFADECAFE)

        Returns:
            32-bit checksum
        """
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
        """
        Calculate ADD16 checksum (algorithm 0x10 = SB_ADD16_ALGO_E).
        Reads 32-bit values, extracts low and high 16-bit words, and adds them.

        ADD16 algorithm:
            lc = *startAdr++;
            chkSum_u32 += (uint16)lc + (uint16)(lc >> 16);

        Args:
            start: Start offset in binary
            end_inclusive: End offset in binary (inclusive, but note: last 4 bytes excluded!)
            initial_value: Initial checksum value (usually 0xFADECAFE)

        Returns:
            32-bit checksum
        """
        if start < 0 or end_inclusive >= len(self.data) or start > end_inclusive:
            return 0

        checksum = initial_value
        pos = start

        # Note: Like ADD32, the last 4 bytes (adjustment value) are NOT included
        # Process 32-bit values, but sum them as two 16-bit words
        while pos <= end_inclusive - 2:
            word = self.data[pos] | (self.data[pos + 1] << 8)
            pos += 2
            checksum = (checksum + word) & 0xFFFFFFFF

        # Last 16-bit word goes into the high 16 bits
        word = self.data[pos] | (self.data[pos + 1] << 8)
        checksum: int = (checksum + (word << 16)) & 0xFFFFFFFF

        return checksum

    def validate_checksum_structure(self, cs: ChecksumStructure, block_start_mem: int,
                                     block_start_bin: int) -> bool:
        """
        Validate a checksum structure by calculating checksum over the specified region.

        Args:
            cs: ChecksumStructure to validate
            block_start_mem: Block start address in memory (e.g., 0x80000000)
            block_start_bin: Block start offset in binary file (e.g., 0x00000000)

        Returns:
            True if checksum is valid, False otherwise
        """
        # Convert memory addresses to binary file offsets
        # Translation: file_offset = memory_address - block_start_mem + block_start_bin
        start_offset = cs.cs_start - block_start_mem + block_start_bin
        end_offset = cs.cs_end - block_start_mem + block_start_bin

        # Validate offsets are within binary
        if start_offset < 0 or end_offset > len(self.data) or start_offset >= end_offset:
            cs.calculated_checksum = None
            cs.is_valid = False
            return False

        # Calculate checksum based on algorithm
        if cs.cs_algorithm == 0x00:
            # Algorithm 0x00: CRC32 (SB_CRC32_ALGO_E)
            # Expected result: 0x35015001 (complement of cs_expected_val)
            checksum = self.calculate_crc32_algo(start_offset, end_offset, cs.cs_start_val)
            cs.calculated_checksum = checksum
            cs.is_valid = (checksum == 0x35015001)
        elif cs.cs_algorithm == 0x01:
            # Algorithm 0x01: ADD32 (SB_ADD32_ALGO_E)
            # Expected result: cs_expected_val directly (0xCAFEAFFE)
            checksum = self.calculate_add32_checksum(start_offset, end_offset, cs.cs_start_val)
            cs.calculated_checksum = checksum
            cs.is_valid = (checksum == cs.cs_expected_val)
        elif cs.cs_algorithm == 0x10:
            # Algorithm 0x10: ADD16 (SB_ADD16_ALGO_E)
            # Expected result: cs_expected_val directly (0xCAFEAFFE)
            checksum = self.calculate_add16_checksum(start_offset, end_offset, cs.cs_start_val)
            cs.calculated_checksum = checksum
            cs.is_valid = (checksum == cs.cs_expected_val)
        else:
            # Unknown algorithm
            cs.calculated_checksum = None
            cs.is_valid = None

        return cs.is_valid if cs.is_valid is not None else False

    def validate_all_checksums(self) -> None:
        """Validate checksums for all checksum structures in all blocks"""
        print("\n[*] Validating checksums...")

        for block in self.bosch_blocks:
            for cs in block.checksum_structures:
                self.validate_checksum_structure(cs, block.block_start, block.bin_start)

        # Count validation results
        total = sum(len(block.checksum_structures) for block in self.bosch_blocks)
        valid = sum(1 for block in self.bosch_blocks
                   for cs in block.checksum_structures if cs.is_valid)

        print(f"[+] Validated {valid}/{total} checksums")

    def _is_flash_addr(self, addr: int) -> bool:
        """Check if address is a valid TriCore flash address."""
        return 0x80000000 <= addr <= 0x807FFFFF

    def find_cvn_config(self) -> Optional[CVNConfig]:
        """
        Find and parse the CVN configuration from the binary.

        The CVN config structure contains pointers to memory regions used
        for calculating the Calibration Verification Number (CRC32).

        Returns:
            CVNConfig if found, None otherwise
        """
        # Find dataset block (0x60)
        ds_block = None
        for block in self.bosch_blocks:
            if block.block_identifier == 0x60:
                ds_block = block
                break

        if not ds_block:
            return None

        ds_start = ds_block.block_start
        ds_end = ds_block.block_end
        base = self.bosch_blocks[0].block_start - self.bosch_blocks[0].bin_start

        # Search for CVN config pattern:
        # { pointer, DS_START, DS_WOCS_END, count }
        for offset in range(0, len(self.data) - 16, 4):
            ptr = self.read_dword_le(offset)
            val1 = self.read_dword_le(offset + 4)
            val2 = self.read_dword_le(offset + 8)
            count = self.read_dword_le(offset + 12)

            # Check pattern: ptr is flash addr, val1 is DS start,
            # val2 is within DS range, count is small
            if (self._is_flash_addr(ptr) and val1 == ds_start and
                self._is_flash_addr(val2) and ds_start < val2 <= ds_end and
                1 <= count <= 4):

                # Follow pointer to get memory section table
                config_offset = ptr - base
                if not (0 <= config_offset < len(self.data)):
                    continue

                memsec_ptr = self.read_dword_le(config_offset)
                memsec_offset = memsec_ptr - base
                if not (0 <= memsec_offset < len(self.data)):
                    continue

                # Read memory section entries (start, end pairs)
                regions = []
                for i in range(4):  # Max 4 sections
                    sec_start = self.read_dword_le(memsec_offset + i * 8)
                    sec_end = self.read_dword_le(memsec_offset + i * 8 + 4)
                    if self._is_flash_addr(sec_start) and self._is_flash_addr(sec_end) and sec_end > sec_start:
                        regions.append((sec_start, sec_end))
                    else:
                        break

                # Add dataset region
                regions.append((val1, val2))

                return CVNConfig(
                    config_offset=offset,
                    regions=regions,
                    ds_start=val1,
                    ds_wocs_end=val2,
                    base_address=base
                )

        return None

    def calculate_cvn(self, data: bytes = None) -> Optional[int]:
        """
        Calculate CVN (Calibration Verification Number) using CRC32.

        The CVN is a CRC32 checksum over specific memory regions defined
        in the CVN configuration.

        Uses a pre-computed lookup table for fast byte-at-a-time processing.
        Processes data as little-endian dwords (4 bytes at a time) to match
        the original TriCore MCU implementation.

        Args:
            data: Optional data to use (defaults to self.data)

        Returns:
            32-bit CVN value, or None if CVN config not found
        """
        if self.cvn_config is None:
            return None

        if data is None:
            data = self.data

        # Initialize lookup table if needed
        init_crc32_table()

        crc = 0xFFFFFFFF
        base = self.cvn_config.base_address

        for mem_start, mem_end in self.cvn_config.regions:
            file_start = mem_start - base
            file_end = mem_end - base

            if file_start < 0 or file_end > len(data):
                continue

            # Process 4 bytes at a time (little-endian dword) using lookup table
            # This matches the original bit-by-bit implementation but is ~8x faster
            pos = file_start
            while pos + 3 <= file_end:
                # Process all 4 bytes of the dword through the CRC
                # In little-endian: byte0, byte1, byte2, byte3
                crc = CRC32_TABLE[(crc ^ data[pos]) & 0xFF] ^ (crc >> 8)
                crc = CRC32_TABLE[(crc ^ data[pos+1]) & 0xFF] ^ (crc >> 8)
                crc = CRC32_TABLE[(crc ^ data[pos+2]) & 0xFF] ^ (crc >> 8)
                crc = CRC32_TABLE[(crc ^ data[pos+3]) & 0xFF] ^ (crc >> 8)
                pos += 4

        return crc ^ 0xFFFFFFFF

    def correct_cvn(self, target_cvn: int, data: bytearray) -> bool:
        """
        Correct CVN to match a target value by patching the dataset region.

        Uses GF(2) matrix solving to find a 4-byte patch value that makes
        the CVN CRC32 equal the target value.

        The patch location is: DS_WOCS_END aligned down to 32-byte boundary

        Args:
            target_cvn: Target CVN value to achieve
            data: Mutable binary data (bytearray)

        Returns:
            True if correction successful, False otherwise
        """
        if self.cvn_config is None:
            return False

        base = self.cvn_config.base_address
        ds_wocs_end_file = self.cvn_config.ds_wocs_end - base

        # Patch location: DS_WOCS_END aligned down to 32-byte boundary
        patch_offset = ds_wocs_end_file & ~0x1F

        # Verify patch is within a CVN region
        patch_in_region = False
        for mem_start, mem_end in self.cvn_config.regions:
            file_start = mem_start - base
            file_end = mem_end - base
            if file_start <= patch_offset < file_end:
                patch_in_region = True
                break

        if not patch_in_region:
            print_error("CVN patch location not within any CVN region")
            return False

        # Use multi-region solver which handles all cases correctly
        return self._correct_cvn_multiregion(target_cvn, data, patch_offset)

    def _correct_cvn_multiregion(self, target_cvn: int, data: bytearray, patch_offset: int) -> bool:
        """
        Correct CVN when patch is in a multi-region calculation.

        Uses GF(2) matrix solving with O(n + log(n)*32^3) complexity:
        1. Calculate CVN with patch=0 (single pass through data)
        2. Compute CRC up to patch point (partial pass, reuses work)
        3. Build local transformation matrix (32 single-dword calculations)
        4. Use matrix exponentiation to propagate effects through remaining data

        This is much faster than computing CVN 33 times.
        """
        init_crc32_table()
        base = self.cvn_config.base_address

        # Find which region contains the patch
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

        # Calculate CVN with patch=0 (needed for final answer)
        data_copy = bytearray(data)
        struct.pack_into('<I', data_copy, patch_offset, 0)
        cvn_with_zero = self.calculate_cvn(bytes(data_copy))

        # Calculate CRC up to (but not including) the patch dword
        crc_to_patch = 0xFFFFFFFF

        # Process all regions before patch region
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

        # Process patch region up to patch offset
        pos = patch_region_file_start
        while pos < patch_offset:
            crc_to_patch = CRC32_TABLE[(crc_to_patch ^ data[pos]) & 0xFF] ^ (crc_to_patch >> 8)
            crc_to_patch = CRC32_TABLE[(crc_to_patch ^ data[pos+1]) & 0xFF] ^ (crc_to_patch >> 8)
            crc_to_patch = CRC32_TABLE[(crc_to_patch ^ data[pos+2]) & 0xFF] ^ (crc_to_patch >> 8)
            crc_to_patch = CRC32_TABLE[(crc_to_patch ^ data[pos+3]) & 0xFF] ^ (crc_to_patch >> 8)
            pos += 4

        # Build local transformation: effect of each patch bit on CRC after patch dword
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

        # Count total CRC bytes after patch (each byte = 1 step through table)
        # The CVN loop uses "while pos + 3 <= file_end", so we need to count
        # how many complete dwords can be processed from patch_offset+4 to file_end
        total_bytes_after = 0

        # Remaining in patch region: count dwords that fit in [patch_offset+4, patch_region_file_end]
        pos = patch_offset + 4
        while pos + 3 <= patch_region_file_end:
            total_bytes_after += 4
            pos += 4

        # Additional regions
        for idx in range(patch_region_idx + 1, len(self.cvn_config.regions)):
            mem_start, mem_end = self.cvn_config.regions[idx]
            file_start = mem_start - base
            file_end = mem_end - base
            if file_start >= 0 and file_end <= len(data):
                pos = file_start
                while pos + 3 <= file_end:
                    total_bytes_after += 4
                    pos += 4

        # Build propagation matrix using matrix exponentiation
        # Single-step matrix: how CRC difference propagates through one table lookup
        # For table CRC: crc_out = TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
        # Difference propagation: d_out = TABLE[d & 0xFF] ^ (d >> 8) when byte is same
        # This is because: (crc1 ^ byte) ^ (crc2 ^ byte) = crc1 ^ crc2 = d
        def diff_step_table(d):
            """Propagate CRC difference through one byte/table step."""
            return CRC32_TABLE[d & 0xFF] ^ (d >> 8)

        # Build single-step matrix
        step_matrix = [diff_step_table(1 << i) for i in range(32)]

        # Matrix operations in GF(2)
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

        # Compute propagation matrix for all bytes after patch
        prop_matrix = matrix_pow_gf2(step_matrix, total_bytes_after)

        # Apply propagation to get final effects on CVN
        patch_effects = []
        for bit in range(32):
            final_effect = apply_matrix(prop_matrix, patch_effects_local[bit])
            patch_effects.append(final_effect)

        target_diff = target_cvn ^ cvn_with_zero

        # Build augmented matrix for Gaussian elimination
        # Each row corresponds to an output bit
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
            # Find pivot
            pivot_row = None
            for row in range(col, 32):
                if (aug_matrix[row][0] >> col) & 1:
                    pivot_row = row
                    break

            if pivot_row is None:
                continue

            # Swap
            if pivot_row != col:
                aug_matrix[col], aug_matrix[pivot_row] = aug_matrix[pivot_row], aug_matrix[col]

            # Eliminate
            for row in range(32):
                if row != col and ((aug_matrix[row][0] >> col) & 1):
                    aug_matrix[row][0] ^= aug_matrix[col][0]
                    aug_matrix[row][1] ^= aug_matrix[col][1]

        # Extract solution
        patch_value = 0
        for row in range(32):
            row_matrix = aug_matrix[row][0]
            target_bit = aug_matrix[row][1]
            if row_matrix == 0:
                if target_bit != 0:
                    return False  # No solution exists
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
        """
        Correct an ADD32 checksum by modifying the last 4 bytes of the checksummed region.

        Args:
            cs: ChecksumStructure to correct
            block_start_mem: Block start address in memory
            block_start_bin: Block start offset in binary file
            data: Mutable binary data (bytearray)

        Returns:
            True if correction successful, False otherwise
        """
        if cs.cs_algorithm != 0x01:
            return False

        # Convert memory addresses to binary file offsets
        start_offset = cs.cs_start - block_start_mem + block_start_bin
        end_offset = cs.cs_end - block_start_mem + block_start_bin

        if start_offset < 0 or end_offset > len(data) or start_offset >= end_offset:
            return False

        # Calculate current checksum
        current_checksum = self.calculate_add32_checksum(start_offset, end_offset, cs.cs_start_val)
        target_checksum = cs.cs_expected_val

        # Calculate difference needed
        difference = (target_checksum - current_checksum) & 0xFFFFFFFF

        # Get last 4 bytes position (end_offset is inclusive, so the last dword starts at end_offset-3)
        last_dword_offset = end_offset - 3
        old_value = struct.unpack('<I', data[last_dword_offset:last_dword_offset+4])[0]

        # Calculate new value: add the difference
        new_value = (old_value + difference) & 0xFFFFFFFF

        # Write new value
        struct.pack_into('<I', data, last_dword_offset, new_value)

        return True

    def correct_add16_checksum(self, cs: ChecksumStructure, block_start_mem: int,
                                block_start_bin: int, data: bytearray) -> bool:
        """
        Correct an ADD16 checksum by modifying the last 4 bytes of the checksummed region.

        ADD16 works similarly to ADD32 but adds 16-bit words instead of 32-bit dwords.
        The correction is tricky because changing the last 4 bytes affects the checksum
        as two 16-bit values.

        Args:
            cs: ChecksumStructure to correct
            block_start_mem: Block start address in memory
            block_start_bin: Block start offset in binary file
            data: Mutable binary data (bytearray)

        Returns:
            True if correction successful, False otherwise
        """
        if cs.cs_algorithm != 0x10:
            return False

        # Convert memory addresses to binary file offsets
        start_offset = cs.cs_start - block_start_mem + block_start_bin
        end_offset = cs.cs_end - block_start_mem + block_start_bin

        if start_offset < 0 or end_offset > len(data) or start_offset >= end_offset:
            return False

        # Calculate current checksum
        current_checksum = self.calculate_add16_checksum(start_offset, end_offset, cs.cs_start_val)
        target_checksum = cs.cs_expected_val

        # Calculate difference needed
        difference = (target_checksum - current_checksum) & 0xFFFFFFFF

        # Get last 4 bytes position (end_offset is inclusive, so the last dword starts at end_offset-3)
        last_dword_offset = end_offset - 3
        old_value = struct.unpack('<I', data[last_dword_offset:last_dword_offset+4])[0]

        # For ADD16, the dword contributes as: low_word + high_word
        # If we change the dword from old_value to new_value:
        # The checksum change is: (new_low + new_high) - (old_low + old_high)
        # We want this to equal difference
        # So: new_low + new_high = old_low + old_high + difference
        # Simple solution: add the difference to the dword value
        # This distributes across both 16-bit words
        new_value = (old_value + difference) & 0xFFFFFFFF

        # Write new value
        struct.pack_into('<I', data, last_dword_offset, new_value)

        return True

    def correct_crc32_checksum(self, cs: ChecksumStructure, block_start_mem: int,
                                block_start_bin: int, block_bin_end: int, data: bytearray) -> bool:
        """
        Correct a CRC32 checksum by forging signature and calculating dCSAdjust.

        Process:
        1. Apply ADD32 corrections first (if needed)
        2. Calculate RIPEMD-160 hash of block (excluding signature + dCSAdjust)
        3. Forge Bleichenbacher RSA signature containing the hash
        4. Write forged signature to binary
        5. Solve for dCSAdjust value that makes CRC32 = 0x35015001

        Args:
            cs: ChecksumStructure to correct
            block_start_mem: Block start address in memory
            block_start_bin: Block start offset in binary file
            block_bin_end: Block end offset in binary file
            data: Mutable binary data (bytearray)

        Returns:
            True if correction successful, False otherwise
        """
        if cs.cs_algorithm != 0x00:
            return False

        # Convert memory addresses to binary file offsets
        start_offset = cs.cs_start - block_start_mem + block_start_bin
        end_offset = cs.cs_end - block_start_mem + block_start_bin

        if start_offset < 0 or end_offset > len(data) or start_offset >= end_offset:
            return False

        # Epilog dCSAdjust is 4 bytes before DEADBEEF
        epilog_adjust_offset = block_bin_end - 7
        signature_offset = epilog_adjust_offset - 128

        # Verify offsets are within checksummed region
        if epilog_adjust_offset < start_offset or epilog_adjust_offset + 3 > end_offset:
            return False

        target_checksum = 0x35015001  # Target for CRC32

        # Calculate RIPEMD-160 hash of block (excluding signature + dCSAdjust)
        hash_start = block_start_bin
        hash_end = signature_offset
        block_data = bytes(data[hash_start:hash_end])

        ripemd160 = hashlib.new('ripemd160')
        ripemd160.update(block_data)
        ripemd_hash = ripemd160.digest()

        # Forge Bleichenbacher signature and write to binary
        forged_signature = forge_bleichenbacher_signature(ripemd_hash)
        data[signature_offset:signature_offset+128] = forged_signature

        # Solve for dCSAdjust value using GF(2) matrix algebra (instant!)
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
        """
        Correct all invalid checksums in the binary.

        Args:
            output_path: Path to write corrected binary. If None, overwrites original.

        Returns:
            Number of checksums corrected
        """
        console.print()
        console.print(Panel("[bold cyan]Checksum Correction Process[/bold cyan]\n" +
                          "Two-pass algorithm: ADD32/ADD16 → CRC32",
                          border_style="cyan"))

        # Create mutable copy of data
        corrected_data = bytearray(self.data)
        corrected_count = 0

        # PASS 1: Correct all ADD32 and ADD16 checksums first
        console.print()
        console.print("[bold blue]PASS 1:[/bold blue] Correcting ADD32 and ADD16 checksums")
        console.print()

        for i, block in enumerate(self.bosch_blocks, 1):
            has_add = any(cs.cs_algorithm in (0x01, 0x10) for cs in block.checksum_structures)
            if not has_add:
                continue

            console.print(f"[yellow]Block {i}:[/yellow] {block.block_name}")

            for j, cs in enumerate(block.checksum_structures, 1):
                if cs.cs_algorithm not in (0x01, 0x10):  # Only ADD32/ADD16 in this pass
                    continue

                # Re-validate with current data
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
                    # Re-validate to confirm
                    self.data = bytes(corrected_data)
                    self.validate_checksum_structure(cs, block.block_start, block.bin_start)
                    if cs.is_valid:
                        print_success(f"Corrected to 0x{cs.calculated_checksum:08X}")
                    else:
                        print_error("Verification failed")
                else:
                    print_error("Correction failed")

        # PASS 2: Correct all CRC32 checksums
        console.print()
        console.print("[bold blue]PASS 2:[/bold blue] Correcting CRC32 checksums")
        console.print()

        for i, block in enumerate(self.bosch_blocks, 1):
            has_crc32 = any(cs.cs_algorithm == 0x00 for cs in block.checksum_structures)
            if not has_crc32:
                continue

            console.print(f"[yellow]Block {i}:[/yellow] {block.block_name}")

            for j, cs in enumerate(block.checksum_structures, 1):
                if cs.cs_algorithm != 0x00:  # Only CRC32 in this pass
                    continue

                # Re-validate with current data (now includes ADD32 corrections!)
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
                    # Re-validate to confirm
                    self.data = bytes(corrected_data)
                    self.validate_checksum_structure(cs, block.block_start, block.bin_start)
                    if cs.is_valid:
                        print_success(f"Corrected to 0x{cs.calculated_checksum:08X}")
                    else:
                        print_error("Verification failed")
                else:
                    print_error("Correction failed")

        # Restore original data
        self.data = bytes(self.data)

        # Write corrected binary if requested
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

    def print_summary(self) -> None:
        """Print comprehensive summary with rich formatting"""
        console.print()

        # File info panel
        file_info = f"[cyan]{self.binary_path.name}[/cyan]\n"
        file_info += f"Size: 0x{len(self.data):X} ({len(self.data):,} bytes)"
        console.print(Panel(file_info, title="📁 Binary File", border_style="cyan"))

        # ECU Variant
        variants = self.identify_ecu_variant()
        if variants:
            variant_text = "\n".join(f"• {v}" for v in variants)
            console.print(Panel(variant_text, title="ECU Variant", border_style="blue"))

        # CVN Info
        if self.cvn_config and self.cvn_config.calculated_cvn is not None:
            cvn_text = f"[bold]CVN:[/bold] 0x{self.cvn_config.calculated_cvn:08X}\n"
            cvn_text += f"[dim]Regions: {len(self.cvn_config.regions)}[/dim]"
            console.print(Panel(cvn_text, title="CVN (Calibration Verification Number)", border_style="magenta"))

        # Bosch Blocks summary
        console.print()
        console.print(f"[bold cyan]═══ Bosch Checksum Blocks ({len(self.bosch_blocks)} found) ═══[/bold cyan]")

        for i, block in enumerate(self.bosch_blocks, 1):
            console.print()

            # Block header with OTP indicator if set
            otp_indicator = " [red][OTP][/red]" if block.has_otp else ""
            header = f"[bold yellow]Block {i}:[/bold yellow] [cyan]{block.block_name}[/cyan]{otp_indicator}"
            console.print(header)

            # Block info table
            info_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
            info_table.add_column("Property", style="dim")
            info_table.add_column("Value")

            info_table.add_row("Location", f"0x{block.bin_start:08X} - 0x{block.bin_end:08X}")
            info_table.add_row("Memory", f"0x{block.block_start:08X} - 0x{block.block_end:08X}")
            info_table.add_row("Size", f"0x{block.size:X} ({block.size:,} bytes)")
            info_table.add_row("Identifier", f"0x{block.block_identifier:08X} (type: 0x{block.block_type_id:02X})")

            console.print(info_table)

            # Checksum structures table
            if block.checksum_structures:
                console.print()
                cs_table = Table(title=f"Checksum Structures ({len(block.checksum_structures)})",
                               box=box.ROUNDED, show_lines=True)

                cs_table.add_column("#", style="dim", width=3)
                cs_table.add_column("Algorithm", width=10)
                cs_table.add_column("Range", width=25)
                cs_table.add_column("Calculated", width=10, justify="right")
                cs_table.add_column("Expected", width=10, justify="right")
                cs_table.add_column("Status", width=8, justify="center")

                for j, cs in enumerate(block.checksum_structures, 1):
                    algo_name = {0x00: "CRC32", 0x01: "ADD32", 0x10: "ADD16"}.get(cs.cs_algorithm, "UNKNOWN")
                    range_str = f"0x{cs.cs_start:08X}\n0x{cs.cs_end:08X}"

                    if cs.calculated_checksum is not None:
                        calc_str = f"0x{cs.calculated_checksum:08X}"
                        exp_str = "0x35015001" if cs.cs_algorithm == 0x00 else "0xCAFEAFFE"

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

        # Find and calculate CVN
        self.cvn_config = self.find_cvn_config()
        if self.cvn_config:
            self.cvn_config.calculated_cvn = self.calculate_cvn()
            print(f"[+] CVN: 0x{self.cvn_config.calculated_cvn:08X}")

        self.print_summary()


def main():
    """Main entry point"""
    import argparse

    # Display banner
    print_banner()

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

    args = parser_args.parse_args()

    # Validate arguments
    if args.correct and args.overwrite and args.output:
        print_error("Cannot specify both --output and --overwrite")
        sys.exit(1)

    import time
    start_time = time.time()

    try:
        parser = MEDC17BinaryParser(args.binary_file)
        parser.parse()

        # Determine output path
        output_path = None
        if args.output:
            output_path = args.output
        elif args.overwrite:
            output_path = args.binary_file

        # Perform correction if requested
        if args.correct or args.fix_cvn:
            if not output_path:
                console.print()
                print_warning("Correction requested but no output path given")
                print_info("Use --output <file> or --overwrite to save corrections")
            else:
                # Load data for corrections
                corrected_data = bytearray(parser.data)

                # Fix CVN first if requested (before checksums, since CVN patch is within checksum range)
                if args.fix_cvn:
                    console.print()
                    console.print(Panel("[bold cyan]CVN Correction[/bold cyan]",
                                      border_style="cyan"))

                    # Load original file and get its CVN
                    if not Path(args.fix_cvn).exists():
                        print_error(f"Original file not found: {args.fix_cvn}")
                        sys.exit(1)

                    original_parser = MEDC17BinaryParser(args.fix_cvn)
                    original_parser.load_binary()
                    original_parser.find_bosch_blocks()
                    original_parser.cvn_config = original_parser.find_cvn_config()

                    if original_parser.cvn_config is None:
                        print_error("Could not find CVN config in original file")
                        sys.exit(1)

                    target_cvn = original_parser.calculate_cvn()
                    print_info(f"Target CVN (from original): 0x{target_cvn:08X}")

                    # Calculate current CVN
                    parser.data = bytes(corrected_data)
                    current_cvn = parser.calculate_cvn()
                    print_info(f"Current CVN: 0x{current_cvn:08X}")

                    if current_cvn == target_cvn:
                        print_success("CVN already matches target")
                    else:
                        # Correct CVN
                        console.print("[yellow]Correcting CVN...[/yellow]")
                        success = parser.correct_cvn(target_cvn, corrected_data)

                        if not success:
                            print_error("CVN correction failed")
                            sys.exit(1)

                        print_success(f"CVN patched for target 0x{target_cvn:08X}")

                # Now correct checksums (handles both --correct and --fix-cvn cases)
                # CVN patch is within checksum range, so checksums need recalculating
                parser.data = bytes(corrected_data)
                parser.correct_all_checksums(output_path)

                # Reload the corrected data
                with open(output_path, 'rb') as f:
                    corrected_data = bytearray(f.read())

                # Verify CVN is still correct after checksum fix (if we did CVN correction)
                if args.fix_cvn:
                    console.print()
                    console.print("[dim]Verifying CVN after checksum correction...[/dim]")
                    parser.data = bytes(corrected_data)
                    new_cvn = parser.calculate_cvn()
                    if new_cvn == target_cvn:
                        print_success(f"CVN verified: 0x{new_cvn:08X}")
                    else:
                        print_error(f"CVN verification failed: got 0x{new_cvn:08X}")

        # Print elapsed time
        elapsed = time.time() - start_time
        console.print()
        console.print(f"[dim]Completed in {elapsed:.2f}s[/dim]")

    except FileNotFoundError as e:
        print_error(f"File not found: {args.binary_file}, {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
