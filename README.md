# MEDC17 Checksum Tool

Checksum analyzer and corrector for Bosch MED17/EDC17 ECU firmware binaries.

## Overview

Analyzes and corrects checksums in Bosch MED17 and EDC17 ECU firmware files. Supports CRC32, ADD32, and ADD16 algorithms.

> [!WARNING]
> Currently tested primarily with calibration changes. Code section modifications may require additional validation. ECUs utilising the "Variant Dataset" may not currently be properly corrected by this tool.

## Features

- ✅ **Automatic block detection** - Identifies all checksum blocks in the binary
- ✅ **Three algorithms** - CRC32, ADD32, and ADD16 support
- ✅ **Instant CRC32 solving** - GF(2) matrix algebra
- ✅ **RSA signature forging** - Generates valid Bleichenbacher signatures
- ✅ **CVN correction** - Fix Calibration Verification Number to match original
- ✅ **Safe operation** - Never overwrites original files

---

<div align="center">

[<img src="./hs-logo.png" width="50%">](https://howellsystems.co.uk)

</div>

### Need Help with Your VW Group ECU Project?

Got a custom ECU tool you want built? Need reverse engineering work done? Working on something ambitious for VW Group control units?

**[Howell Systems Ltd](https://howellsystems.co.uk)** provides professional reverse engineering and custom embedded software development for the automotive aftermarket. 

**[Get in touch →](https://howellsystems.co.uk)**



---

## Requirements

Python 3.7+ with `rich` package:
```bash
pip install rich
```

## Usage

### Analyze checksums (read-only)
```bash
python main.py firmware.bin
```

### Correct checksums
```bash
python main.py firmware.bin --correct --output firmware_corrected.bin
```

### Correct checksums and CVN
```bash
python main.py modified.bin --correct --fix-cvn original.bin --output fixed.bin
```

### Correct checksums and preserve CVN without the original file
```bash
python main.py modified.bin --correct --fix-cvn-inplace --output fixed.bin
```

### Options
```
  -h, --help            Show help
  -c, --correct         Correct invalid checksums
  -o OUTPUT, --output   Output file path for corrected binary
  --fix-cvn ORIGINAL    Fix CVN to match the CVN from ORIGINAL file
  --fix-cvn-inplace     Preserve CVN without the original file (best effort)
```

## How It Works

### Checksum Block Structure
ECU firmware contains multiple Bosch blocks with checksums protecting different regions:
- **Absolute constants** (0xC0)
- **Customer Block** (0x30)
- **Application software** (0x40)
- **Dataset** (0x60)
- **Startup Block** (0x10)

### Two-Pass Correction
1. **Pass 1 (ADD32/ADD16)**: Modifies last 4 bytes of checksummed regions using simple arithmetic
2. **Pass 2 (CRC32)**: Uses GF(2) matrix algebra to solve for exact dCSAdjust value instantly

### CRC32 Mathematical Solving
Traditional methods brute-force billions of values. This tool models CRC32 as 32 linear equations over GF(2), constructs a 32×32 matrix, and solves using Gaussian elimination.

### CVN (Calibration Verification Number)
The CVN is a CRC32 checksum over specific memory regions that OBD-II diagnostics report to verify calibration integrity. When modifying calibration data, the CVN changes and will no longer match dealer records.

**How CVN correction works:**

Instead of hardcoding regions per ECU variant, this tool dynamically discovers the CVN configuration structure directly from the binary:

1. **Structure Discovery** - Locates the CVN config by scanning for patterns (we always know that DS0 will appear in the main struct)
2. **Region Extraction** - Reads the actual memory regions from the structure (typically ASW + Dataset areas, but this can vary by ECU)
3. **Patch Location** - Calculates an optimal patch point within the CVN range but outside checksum-critical areas
4. **GF(2) Solving** - Uses matrix algebra to find a 4-byte patch value that produces the target CVN

This approach works across different MED17/EDC17 variants without requiring variant-specific configuration.

**Future enhancement:** A CSV database of software version numbers and their corresponding CVN values could eliminate the need for the original binary file entirely.

## Technical Details

| Algorithm | Initial Value | Expected Result | Method |
|-----------|---------------|-----------------|---------|
| **CRC32** | 0xFADECAFE | 0x35015001 | IEEE 802.3 (0xEDB88320) bit-reversed polynomial |
| **ADD32** | 0xFADECAFE | 0xCAFEAFFE | 32-bit dword sum (overflow ignored) |
| **ADD16** | 0xFADECAFE | 0xCAFEAFFE | Sum of 16-bit words from 32-bit dwords |

## TODO

Future enhancements planned:

- [ ] **Sync blocks** - referenced by some tools, what are they??
- [ ] **ECM/Code monitoring checksums** - Make sure all monitoring checksums are corrected
- [ ] **Variant Dataset Correction** - Even with just calibration changes the VDS block epilog may need changes if it exists, needs investigation
- [ ] **Extended testing** - Validation with code section modifications beyond calibration changes

Contributions welcome!

## License

MIT License - see [LICENSE](LICENSE)

## Credits

Copyright (c) 2025 Connor Howell

## Disclaimer

> [!NOTE]
> For educational and research purposes. Use responsibly on firmware you own or have permission to modify. This implementation may not produce identical output to commercial tools.
