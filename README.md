# MEDC17 Checksum Tool

Checksum analyzer and corrector for Bosch MED17/EDC17 ECU firmware binaries.

Analyzes and corrects checksums in Bosch MED17 and EDC17 ECU firmware files, supporting CRC32, ADD32, and ADD16 algorithms.

> [!WARNING]
> Currently tested primarily with calibration changes. Code section modifications may require additional validation.

## Features

- ✅ **Automatic block detection** - Identifies all checksum blocks in the binary
- ✅ **Three algorithms** - CRC32, ADD32, and ADD16 support
- ✅ **Instant CRC32 solving** - GF(2) matrix algebra instead of brute force
- ✅ **RSA signature forging** - Generates valid Bleichenbacher signatures
- ✅ **CVN correction** - Fixes the Calibration Verification Number to match the original
- ✅ **Multi-variant support** - Corrects the monitoring checksum for every calibration variant
- ✅ **Safe operation** - Never overwrites original files unless you ask it to

---

<div align="center">

[<img src="./hs-logo.png" width="50%">](https://howellsystems.co.uk)

</div>

### Need help with your VW Group ECU project?

Got a custom ECU tool you want built? Need reverse engineering work done? Working on something ambitious for VW Group control units?

**[Howell Systems Ltd](https://howellsystems.co.uk)** provides professional reverse engineering and custom embedded software development for the automotive aftermarket.

If you'd rather not run this locally, the same correction is available as a web tool at **[howellsystems.co.uk/tools](https://howellsystems.co.uk/tools)** — upload a binary, see what's invalid before you commit to anything, and download the corrected file. The other tools I've built around these ECUs live there too: original file extraction from FRF/ODX/PDX containers, MG1/MD1 checksum correction, and DTC deletion for EDC17/MED17.

**[Get in touch →](https://howellsystems.co.uk)**

---

## Requirements

Python 3.7+. The `rich` package is optional — it only pretties up the report, and `--json` never needs it:

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

### Machine-readable output
```bash
python main.py firmware.bin --json
```

Writes a single JSON document to stdout — blocks, every checksum with its calculated and expected value, CVN status, and whether the CVN can be preserved — and nothing else. Exit code `0` means all checksums are valid (or every correction succeeded), `1` means invalid checksums, `2` an error.

### Options
```
  -h, --help            Show help
  -c, --correct         Correct invalid checksums
  -o, --output FILE     Output file path for the corrected binary
  --overwrite           Overwrite the input file in place (dangerous!)
  --fix-cvn ORIGINAL    Fix CVN to match the CVN from the ORIGINAL file
  --fix-cvn-inplace     Preserve CVN without the original file (best effort)
  --json                Emit one JSON document on stdout and nothing else
```

## How it works

### Checksum block structure
ECU firmware contains multiple Bosch blocks with checksums protecting different regions:

- **Absolute constants** (0xC0)
- **Customer Block** (0x30)
- **Application software** (0x40)
- **Dataset** (0x60)
- **Startup Block** (0x10)

### Two-pass correction
1. **Pass 1 (ADD32/ADD16)**: Modifies the last 4 bytes of checksummed regions using simple arithmetic
2. **Pass 2 (CRC32)**: Uses GF(2) matrix algebra to solve for the exact `dCSAdjust` value instantly

### CRC32 mathematical solving
Traditional methods brute-force billions of values. This tool models CRC32 as 32 linear equations over GF(2), constructs a 32×32 matrix, and solves it using Gaussian elimination.

### Multi-variant calibrations
Some ECUs carry more than one calibration variant. The variant dataset block (0x80) holds a table of parameter addresses per variant: most entries are shared, but where a variant overrides a parameter its entry points at a private copy instead.

The monitoring checksum is calculated over whichever variant is active, so each one needs its own compensation value. Correcting only the first leaves the rest wrong, and an ECU running a non-default variant will reject the file — typically as a boot loop.

The tool reads the variant count from the block header, locates the base address table and its per-variant copies, then for each variant resolves every read through that variant's table and writes its compensation value to the matching address.

### CVN (Calibration Verification Number)
The CVN is a CRC32 checksum over specific memory regions that OBD-II diagnostics report to verify calibration integrity. When you modify calibration data, the CVN changes and no longer matches dealer records.

Instead of hardcoding regions per ECU variant, this tool discovers the CVN configuration structure directly from the binary:

1. **Structure discovery** - Locates the CVN config by scanning for patterns (DS0 always appears in the main struct)
2. **Region extraction** - Reads the actual memory regions from the structure (typically ASW + Dataset areas, but this can vary by ECU)
3. **Patch location** - Calculates an optimal patch point within the CVN range but outside checksum-critical areas
4. **GF(2) solving** - Uses matrix algebra to find a 4-byte patch value that produces the target CVN

This approach works across different MED17/EDC17 variants without variant-specific configuration.

> [!NOTE]
> **Future enhancement:** a CSV database of software version numbers and their corresponding CVN values could eliminate the need for the original binary file entirely.

## Technical details

| Algorithm | Initial Value | Expected Result | Method |
|-----------|---------------|-----------------|--------|
| **CRC32** | 0xFADECAFE | 0x35015001 | IEEE 802.3 (0xEDB88320) bit-reversed polynomial |
| **ADD32** | 0xFADECAFE | 0xCAFEAFFE | 32-bit dword sum (overflow ignored) |
| **ADD16** | 0xFADECAFE | 0xCAFEAFFE | Sum of 16-bit words from 32-bit dwords — the final word is added into the high half, so the region's last dword counts as a full 32-bit value |

## Roadmap

- [x] **ECM/code monitoring checksums** - the monitoring checksum is the per-variant one, now corrected
- [x] **Variant dataset correction** - implemented; see [Multi-variant calibrations](#multi-variant-calibrations)
- [ ] **Sync blocks** - a commercial-vehicle feature: some ECUs carry a duplicate control block at a second address range, and a corrected block is copied across so the two stay in sync. Not implemented. At least one EDC17CV52 layout is a double block that must *not* be copied.
- [ ] **Extended testing** - validation with code section modifications beyond calibration changes
- [ ] **Variant coverage** - the variant path is so far exercised against a single multi-variant file

Contributions welcome!

## License

MIT License - see [LICENSE](LICENSE)

Copyright (c) 2025 Connor Howell.

## Disclaimer

> [!NOTE]
> For educational and research purposes. Use responsibly on firmware you own or have permission to modify. This implementation may not produce identical output to commercial tools.
