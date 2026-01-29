#!/bin/bash
#
# Compile all .mlpackage files in gemma3_sweep to .mlmodelc
#

set -e

INPUT_DIR="/Volumes/Models/ANE/gemma3_sweep"

echo ""
echo "=========================================="
echo "Compiling all .mlpackage files"
echo "=========================================="
echo "Input: $INPUT_DIR"
echo ""

# Count total
TOTAL=$(ls -d "$INPUT_DIR"/*.mlpackage 2>/dev/null | wc -l | tr -d ' ')
echo "Found $TOTAL .mlpackage files"
echo ""

COUNT=0
for pkg in "$INPUT_DIR"/*.mlpackage; do
    if [ -d "$pkg" ]; then
        COUNT=$((COUNT + 1))
        basename=$(basename "$pkg" .mlpackage)
        output="$INPUT_DIR/${basename}.mlmodelc"

        echo "[$COUNT/$TOTAL] Compiling: $basename"

        if [ -d "$output" ]; then
            echo "  Skipping (already exists): $output"
        else
            xcrun coremlcompiler compile "$pkg" "$INPUT_DIR/"
            echo "  Done: $output"
        fi
    fi
done

echo ""
echo "=========================================="
echo "COMPILATION COMPLETE"
echo "=========================================="
echo ""
echo "Compiled models:"
ls -la "$INPUT_DIR"/*.mlmodelc 2>/dev/null | head -20 || echo "No .mlmodelc files found"
echo ""
