import sys


def parse_input(text):
    act_vals = None
    weight_rows = {}
    bias_vals = None

    for line in text.strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "act":
            act_vals = parts[1:]
        elif parts[0] == "wgt":
            row_idx = int(parts[1])
            weight_rows[row_idx] = parts[2:]
        elif parts[0] == "bias":
            bias_vals = parts[1:]

    return act_vals, weight_rows, bias_vals


def parse_expected(text):
    results = {}
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        # columns: index, "bias", idx, hex_expected, hex_actual, ...
        try:
            idx = int(parts[2])
            hex_val = parts[3]  # fourth column (0-indexed: col 3)
            results[idx] = hex_val
        except (IndexError, ValueError):
            continue
    return results


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        print("Paste activations, weights, and biases, then enter a blank line:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        text = "\n".join(lines)

    act_vals, weight_rows, bias_vals = parse_input(text)

    if act_vals:
        hex_vals = ", ".join(f"0x{v}" for v in act_vals)
        print(f"uint8_t act_vals[{len(act_vals)}] = {{{hex_vals}}};")
        print()

    num_rows = max(weight_rows.keys()) + 1 if weight_rows else 0
    print(f"uint8_t weight_vals_row[{num_rows}][32] = {{")
    for i in range(num_rows):
        row = weight_rows.get(i, [])
        hex_vals = ", ".join(f"0x{v}" for v in row)
        comma = "," if i < num_rows - 1 else ""
        print(f"    {{{hex_vals}}}{comma}  // row {i}")
    print("};")

    if bias_vals:
        print()
        hex_vals = ", ".join(f"0x{v}" for v in bias_vals)
        print(f"uint8_t bias[{len(bias_vals)}] = {{{hex_vals}}};")

    print()
    print("Paste expected results, then enter a blank line:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    expected_text = "\n".join(lines)

    expected = parse_expected(expected_text)
    if expected:
        num_results = max(expected.keys()) + 1
        hex_vals = ", ".join(expected.get(i, "0x0000") for i in range(num_results))
        print()
        print(f"uint16_t expected_results[{num_results}] = {{{hex_vals}}};")


if __name__ == "__main__":
    main()
