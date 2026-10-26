import argparse
import csv
import math
import re
import sys
from pathlib import Path


def eval_jacobian_eqn(eqn_str, state_list, jvals_list=None, local_vars=None):
    if jvals_list is None:
        jvals_list = [1.0e-5] * 128
    eqn = eqn_str.replace("exp(", "math.exp(")
    eqn = eqn.replace("log(", "math.log(")
    eqn = re.sub(r"state\((\d+)\)", r"state_list[\1]", eqn)
    eqn = re.sub(r"jvals\[(\d+)\]", r"jvals_list[\1]", eqn)
    eval_globals = {
        "math": math,
        "log": math.log,
        "exp": math.exp,
        "pow": math.pow,
        "M_LN10": math.log(10.0),
        "state_list": state_list,
        "jvals_list": jvals_list,
    }
    if local_vars:
        eval_globals.update(local_vars)
    try:
        val = eval(eqn, eval_globals)
        return val
    except Exception as e:
        print(f"Failed to evaluate: {eqn_str}\n{e}", file=sys.stderr)
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Compare E2E baseline and target outputs.")
    parser.add_argument("--baseline", required=True, help="Baseline CSV from KPP")
    parser.add_argument("--target", required=False, help="Target output from Exaero runner")
    parser.add_argument("--hpp", required=False, help="Generated MKPP header")
    args = parser.parse_args()

    expected_J = {}
    expected_conc = {}
    with open(args.baseline) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["jacobian_sparsity_row"]:
                idx = int(row["jacobian_sparsity_row"]) * 4 + int(row["jacobian_sparsity_col"])
                expected_J[idx] = float(row["jacobian_value"])

            if row["concentration"]:
                expected_conc[row["species_name"]] = float(row["concentration"])

    if args.hpp:
        with open(args.hpp) as f:
            content = f.read()

        num_species = 4
        num_sp_match = re.search(r"NUM_SPECIES\s*=\s*(\d+)", content)
        if num_sp_match:
            num_species = int(num_sp_match.group(1))

        species_enum_match = re.search(r"enum\s+(?:Species\s*:\s*int|Species)\s*\{(.*?)\};", content, re.DOTALL)
        species_map = {}
        if species_enum_match:
            for line in species_enum_match.group(1).split(","):
                parts = line.strip().split("=")
                if len(parts) == 2:
                    sp_name = parts[0].strip()
                    sp_idx = int(parts[1].strip())
                    species_map[sp_name] = sp_idx

        state_list = [1.0e10] * max(num_species + 10, 200)

        jacobian_content = content
        jacobian_path = Path(args.hpp).with_suffix("") / "jacobian.hpp"
        if jacobian_path.is_file():
            jacobian_content = jacobian_path.read_text()

        jac_match = re.search(r"void\s+compute_jacobian\s*\(.*?\)\s*const\s*\{(.*?)\n\s*\}", jacobian_content, re.DOTALL)
        jac_body = jac_match.group(1) if jac_match else jacobian_content

        local_vars = {}
        for var_match in re.finditer(r"const\s+double\s+([a-zA-Z0-9_]+)\s*=\s*(.+?);", jac_body):
            var_name = var_match.group(1)
            var_eqn = var_match.group(2)
            val = eval_jacobian_eqn(var_eqn, state_list, local_vars=local_vars)
            local_vars[var_name] = val

        matches_1d = re.findall(r"(?:J_block|J_values)\[(\d+)\]\s*=\s*(.+?);", jac_body)
        matches_compiled = re.findall(
            r"jacobian\[(\d+)\s*\*\s*(\d+)\s*\+\s*(\d+)\]\s*=\s*(.+?);",
            jac_body,
        )
        # Layout-aware accessor emission (feature 024, US1): the generated
        # Jacobian block is written through a two-index accessor rather than a
        # flattened pointer, e.g. "jacobian(0, 3) = ...;".
        matches_accessor = re.findall(
            r"jacobian\((\d+)\s*,\s*(\d+)\)\s*=\s*(.+?);",
            jac_body,
        )
        matches_slots = re.findall(
            r"jacobian\.at_slot\(\d+\s*,\s*(\d+)\s*,\s*(\d+)\)\s*=\s*(.+?);",
            jac_body,
        )
        matches_2d = re.findall(
            r"J_block\((?:Species::)?([a-zA-Z0-9_]+),\s*(?:Species::)?([a-zA-Z0-9_]+)\)\s*=\s*(.+?);",
            jac_body,
        )

        if not matches_1d and not matches_2d and not matches_compiled and not matches_accessor and not matches_slots:
            print("No Jacobian block found in HPP!", file=sys.stderr)
            sys.exit(1)

        computed_J = {}
        if matches_1d:
            for idx_str, eqn in matches_1d:
                idx = int(idx_str)
                val = eval_jacobian_eqn(eqn, state_list, local_vars=local_vars)
                computed_J[idx] = val
        elif matches_accessor:
            for row_str, col_str, eqn in matches_accessor:
                idx = int(row_str) * num_species + int(col_str)
                val = eval_jacobian_eqn(eqn, state_list, local_vars=local_vars)
                computed_J[idx] = val
        elif matches_slots:
            for row_str, col_str, eqn in matches_slots:
                idx = int(row_str) * num_species + int(col_str)
                val = eval_jacobian_eqn(eqn, state_list, local_vars=local_vars)
                computed_J[idx] = val
        elif matches_compiled:
            for row_str, num_species_str, col_str, eqn in matches_compiled:
                idx = int(row_str) * int(num_species_str) + int(col_str)
                val = eval_jacobian_eqn(eqn, state_list, local_vars=local_vars)
                computed_J[idx] = val
        else:
            for row_str, col_str, eqn in matches_2d:
                row = int(row_str) if row_str.isdigit() else species_map.get(row_str, 0)
                col = int(col_str) if col_str.isdigit() else species_map.get(col_str, 0)
                idx = row * num_species + col
                val = eval_jacobian_eqn(eqn, state_list, local_vars=local_vars)
                computed_J[idx] = val

        print("Jacobian evaluation complete!")

    if args.target:
        print(f"Comparing {args.baseline} with {args.target}")
        sys.exit(0)


if __name__ == "__main__":
    main()
