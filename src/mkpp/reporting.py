from pathlib import Path

import networkx as nx

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def write_report(mech, sympy_meta, out_dir, suffix=""):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Graph Generation
    if HAS_MATPLOTLIB:
        G = nx.DiGraph()
        for r in mech.reactions:
            for reactant in r.reactants.keys():
                for product in r.products.keys():
                    if reactant != product:
                        G.add_edge(reactant, product)

        plt.figure(figsize=(12, 10))
        # Use spring layout for better separation
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=0)

        # Calculate degree centrality to size nodes
        degrees = dict(G.degree())
        node_sizes = [v * 100 for v in degrees.values()]

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

        plt.title(f"Chemical Mechanism Topology: {mech.name}")
        plt.axis("off")

        graph_path = out_path / f"network_graph_{mech.name}{suffix}.png"
        plt.savefig(graph_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print(f"Notice: matplotlib is not installed. Skipping network graph generation for {mech.name}.")

    # 2. Stiffness Analytics
    J = sympy_meta.get("jacobian_matrix")
    if J:
        # Count non-zeros per row (species)
        species_list = sympy_meta["species_map"]
        row_density = []
        for i in range(J.shape[0]):
            nz = sum([1 for j in range(J.shape[1]) if J[i, j] != 0])
            row_density.append((species_list[i], nz))

        row_density.sort(key=lambda x: x[1], reverse=True)
        top_dense = row_density[:5]
    else:
        top_dense = []

    # Count reaction types
    type_counts = {}
    stiff_count = 0
    non_stiff_count = 0
    for r in mech.reactions:
        type_counts[r.reaction_type] = type_counts.get(r.reaction_type, 0) + 1
        if r.stiff:
            stiff_count += 1
        else:
            non_stiff_count += 1

    # 3. Write Markdown Report
    report_path = out_path / f"report_{mech.name}{suffix}.md"
    with open(report_path, "w") as f:
        f.write(f"# MKPP Mechanism Diagnostic Report: {mech.name}\n\n")

        f.write("## Overview\n")
        f.write(f"- **Total Species**: {len(mech.species)}\n")
        f.write(f"- **Total Reactions**: {len(mech.reactions)}\n\n")

        f.write("### Reaction Types Breakdown\n")
        for rt, cnt in type_counts.items():
            f.write(f"- **{rt}**: {cnt}\n")
        f.write("\n")

        f.write("### Stiffness Partitioning\n")
        f.write(f"- **Implicit (Stiff) Reactions**: {stiff_count}\n")
        f.write(f"- **Explicit (Non-Stiff) Reactions**: {non_stiff_count}\n")

        if hasattr(mech, "partition_metadata") and mech.partition_metadata:
            f.write("- **Graph Topology Status**: Mechanism contains cyclically dependent fast radicals. Tarjan SCC was applied.\n")
        f.write("\n")

        f.write("## Generated SymPy Rate Expressions\n")
        f.write("Below are the exact algebraic AST expressions evaluated by SymPy for the Unified Jacobian.\n\n")
        f.write("```text\n")
        if hasattr(mech, "sympy_reactions_strings"):
            for i, r_str in enumerate(mech.sympy_reactions_strings):
                f.write(f"R{i+1}: {r_str}\n")
        else:
            for i, r in enumerate(mech.reactions):
                f.write(
                    f"R{i+1}: {r.reaction_type} | Reactants: {list(r.reactants.keys())} | Products: {list(r.products.keys())} | A: {r.parameters.get('A', 'N/A')}\n"
                )

        if sympy_meta and "f_implicit" in sympy_meta:
            F_imp = sympy_meta["f_implicit"]
            F_exp = sympy_meta["f_explicit"]
            species_list = sympy_meta["species_map"]
            for i in range(len(F_imp)):
                f.write(f"d[{species_list[i]}]/dt = (IMPLICIT: {F_imp[i]}) + (EXPLICIT: {F_exp[i]})\n")
        elif sympy_meta and "f_vector" in sympy_meta:
            F = sympy_meta["f_vector"]
            species_list = sympy_meta["species_map"]
            for i in range(len(F)):
                f.write(f"d[{species_list[i]}]/dt = {F[i]}\n")
        f.write("```\n\n")

        f.write("## Performance & Stiffness Diagnostics\n")
        if top_dense:
            f.write(
                "The following species are the most heavily coupled (densest Jacobian rows). These dictate the performance ceiling of the Dense LU / ROS2 implicit solver block:\n\n"
            )
            f.write("| Species | Non-Zero Dependencies |\n")
            f.write("|---------|-----------------------|\n")
            for sp, nz in top_dense:
                f.write(f"| {sp} | {nz} |\n")
        f.write("\n")

        f.write("### Warnings\n")
        warnings = []
        if len(mech.species) > 50:
            warnings.append("Mechanism exceeds 50 species.")
        if type_counts.get("TROE", 0) > 0 or type_counts.get("EP2", 0) > 0 or type_counts.get("EP3", 0) > 0:
            warnings.append(
                "Mechanism contains complex pressure-dependent or empirical falloff rates which expand the AST depth significantly."
            )

        if warnings:
            for w in warnings:
                f.write(f"- ⚠️ {w}\n")
        else:
            f.write("No major warnings.\n")

    print(f"Report emitted to {report_path}")
