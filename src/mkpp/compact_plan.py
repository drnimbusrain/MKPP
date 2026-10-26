"""Build compact numeric storage and ordered sparse LU traversal metadata."""

from typing import Any


def compact_plan(
    species_count: int,
    jacobian_entries: list[tuple[int, int, str]],
    permutation: list[int] | None,
    l_locations: set[tuple[int, int]],
    u_locations: set[tuple[int, int]],
) -> dict[str, Any]:
    """Preserve row-wise Doolittle operation order while omitting structural zeros."""
    n = species_count
    order = permutation or list(range(n))
    if n <= 0 or len(order) != n or set(order) != set(range(n)):
        raise ValueError("FATAL ERROR: compact plan requires a complete species permutation")
    if any((i, i) not in u_locations for i in range(n)):
        raise ValueError("FATAL ERROR: compact LU plan is missing a diagonal pivot")
    if any(not (0 <= j < i < n) for i, j in l_locations) or any(not (0 <= i <= j < n) for i, j in u_locations):
        raise ValueError("FATAL ERROR: compact LU entries are outside their triangular structure")
    physical_j = [(i, j) for i, j, _ in jacobian_entries]
    inverse = {physical: logical for logical, physical in enumerate(order)}
    j_slots = {(inverse[i], inverse[j]): slot for slot, (i, j) in enumerate(physical_j)}
    w_locations = sorted(set(j_slots) | {(i, i) for i in range(n)})
    w_slots = {location: slot for slot, location in enumerate(w_locations)}
    lu_slots = {location: slot for slot, location in enumerate(sorted(l_locations | u_locations))}
    operations = []
    products = []
    for row in range(n):
        locations = [(row, column) for column in range(row, n) if (row, column) in u_locations]
        locations += [(lower, row) for lower in range(row + 1, n) if (lower, row) in l_locations]
        for i, j in locations:
            stop = row
            begin = len(products)
            for inner in range(stop):
                if (i, inner) in l_locations and (inner, j) in u_locations:
                    products.append((lu_slots[i, inner], lu_slots[inner, j]))
            operations.append(
                {
                    "output": lu_slots[i, j],
                    "w": w_slots.get((i, j), -1),
                    "begin": begin,
                    "end": len(products),
                    "diagonal": lu_slots[row, row] if i != row else -1,
                }
            )
    lower = []
    upper = []
    lower_offsets = [0]
    upper_offsets = [0]
    for row in range(n):
        lower.extend((column, lu_slots[row, column]) for column in range(row) if (row, column) in l_locations)
        upper.extend((column, lu_slots[row, column]) for column in range(row + 1, n) if (row, column) in u_locations)
        lower_offsets.append(len(lower))
        upper_offsets.append(len(upper))
    return {
        "j_slots": j_slots,
        "w_locations": w_locations,
        "w_slots": w_slots,
        "lu_slots": lu_slots,
        "operations": operations,
        "products": products,
        "lower": lower,
        "upper": upper,
        "lower_offsets": lower_offsets,
        "upper_offsets": upper_offsets,
        "diagonals": [lu_slots[i, i] for i in range(n)],
        "numeric_bytes": 8 * (max(1, len(j_slots)) + len(w_slots) + len(lu_slots)),
    }
