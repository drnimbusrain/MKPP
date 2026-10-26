from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


@dataclass
class CompilationError(Exception):
    """Structured error for compilation pipeline failures."""

    stage: str  # "parsing", "validation", "lowering", "codegen"
    message: str
    reaction_index: int | None = None
    species_name: str | None = None
    yaml_location: str | None = None  # "file.yaml:line:col"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of the error fields."""
        d: dict[str, Any] = {"stage": self.stage, "message": self.message}
        if self.reaction_index is not None:
            d["reaction_index"] = self.reaction_index
        if self.species_name is not None:
            d["species_name"] = self.species_name
        if self.yaml_location is not None:
            d["yaml_location"] = self.yaml_location
        return d

    def __str__(self) -> str:
        parts = [f"[{self.stage}] {self.message}"]
        if self.reaction_index is not None:
            parts.append(f"reaction_index={self.reaction_index}")
        if self.species_name is not None:
            parts.append(f"species={self.species_name}")
        if self.yaml_location is not None:
            parts.append(f"at {self.yaml_location}")
        return " | ".join(parts)


class PhaseMode(Enum):
    GAS = "gas"
    AEROSOL = "aerosol"
    AQUEOUS = "aqueous"
    MIXED = "mixed"


class SolverMode(Enum):
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    COUPLED = "coupled"


class AerosolRepresentation(Enum):
    BULK = "bulk"
    MODAL = "modal"
    SECTIONAL = "sectional"


class ReactionType(Enum):
    ARRHENIUS = "ARRHENIUS"
    TROE = "TROE"
    PHOTOLYSIS = "PHOTOLYSIS"
    HETEROGENEOUS = "HETEROGENEOUS"
    TUNNELING = "TUNNELING"
    CONDENSATION = "CONDENSATION"
    PHASE_CHANGE = "PHASE_CHANGE"
    EQUILIBRIUM = "EQUILIBRIUM"
    UNKNOWN = "UNKNOWN"


class EquilibriumInput(Enum):
    """Index constants for equilibrium input arrays in generated code."""

    Temperature = 0
    RelativeHumidity = 1
    CloudLiquidWater = 2


@dataclass
class EquilibriumDefinition:
    """Parsed EQUILIBRIUM reaction block from mechanism YAML."""

    system: str  # e.g. "NH4_NO3_SO4"
    total_species: dict[str, list[str]]  # element -> [gas_species, aerosol_species, ...]
    regime_blending: str = "sigmoid"  # "sigmoid"
    transition_width: float = 0.05
    activity_model: str = "fixed"  # "fixed" | "parameterized"
    equilibrium_constants: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {A, dH, Tref} van't Hoff params
    continuous_transition: bool = True
    relaxation_timescale_inv: float = 1e6  # 1/tau [s^-1], default: 1e6


@dataclass
class EquilibriumSymbolicResult:
    """Result of symbolic lowering for an equilibrium system."""

    partition_exprs: dict[str, Any] = field(default_factory=dict)  # species_name -> SymPy Expr: f(C_total, T, RH)
    jacobian_entries: list[tuple[int, int, Any]] = field(default_factory=list)  # (i, j, ∂f_i/∂C_j)
    total_species_map: dict[str, list[int]] = field(default_factory=dict)  # element -> species indices
    regime_weights: list[Any] = field(default_factory=list)  # blending weight expressions
    equilibrium_constants: dict[str, Any] = field(default_factory=dict)  # Keq(T) expressions


@dataclass
class SpeciesDefinition:
    name: str
    phase: PhaseMode
    elements: dict[str, int] = field(default_factory=dict)
    role: str | None = None
    solver_atol: float | None = field(default=None)
    solver_rtol: float | None = field(default=None)


@dataclass
class PhaseDefinition:
    name: str
    solver_mode: SolverMode
    transport_model: str | None = None
    continuity_model: str | None = None


@dataclass
class ReactionDefinition:
    reaction_type: str  # e.g., ARRHENIUS, PHOTOLYSIS, TROE, EP2, EP3
    reactants: dict[str, float]
    products: dict[str, float]
    rate_expression: str
    parameters: dict[str, Any] = field(default_factory=dict)
    unit_policy: str = "canonical"
    stiff: bool = False
    continuous_transition: bool = False


@dataclass
class ArrayDefinition:
    name: str
    rank: int
    layout: str = "LayoutLeft"  # Default to Fortran-compatible column-major
    extent: list[int] | None = None
    unit: str = "unknown"
    ownership: str = "host"  # 'host' or 'device'


@dataclass
class HostInterfaceSchema:
    arrays: list[ArrayDefinition] = field(default_factory=list)


@dataclass
class MechanismDefinition:
    name: str
    description: str
    aerosol_representation: AerosolRepresentation
    species: list[SpeciesDefinition]
    phases: list[PhaseDefinition]
    reactions: list[ReactionDefinition]
    host_interface: Optional["HostInterfaceSchema"] = None
    units: str = "canonical"
    metadata: dict[str, str] = field(default_factory=dict)
    equilibrium_reactions: list[EquilibriumDefinition] = field(default_factory=list)
    # True when any kinetic reaction declares an `activation_trigger` naming a
    # runtime meteorological quantity (e.g. `meteo.cloud_liquid_water > 1e-6`).
    # Such a reaction's rate is gated by a smooth indicator of that quantity, so
    # the generated solver must accept it as an additional per-cell equilibrium
    # input (see EquilibriumInput.CloudLiquidWater). When False, the generated
    # code is byte-identical to a mechanism without triggers.
    has_cloud_gated_reaction: bool = False
    concentration_unit: str | None = None
    kinetic_units_converted_to_si: bool = False


@dataclass
class EnvironmentDefinition:
    temperature: float = 298.15
    pressure: float = 101325.0
    air_density: float = 2.46e19
    relative_humidity: float = 0.5
    solver_atol: float | None = None
    solver_rtol: float | None = None
    solver_initial_step_seed: float | None = None
    initial_concentrations: dict[str, float] = field(default_factory=dict)


@dataclass
class TestEnvironmentDefinition:
    name: str
    initial_conditions: dict[str, float]
    meteorology: dict[str, float]
    photolysis: dict[str, float]
    cloud_liquid_water: float
    bounds: dict[str, dict[str, float]]
    expected_outputs: dict[str, float] | None = None


@dataclass
class GeneratedArtifact:
    kind: str
    path: str
    dependencies: list[str]
    checksum: str | None = None


@dataclass
class SymbolicLUPlan:
    num_species: int
    species_map: list[str]
    non_zero_jacobian: list[tuple[int, int, str]] = field(default_factory=list)
    l_expressions: list[tuple[int, int, str]] = field(default_factory=list)
    u_expressions: list[tuple[int, int, str]] = field(default_factory=list)
    lu_expressions_ordered: list[tuple[str, int, int, str]] = field(default_factory=list)
    forward_sub_steps: list[tuple[int, str]] = field(default_factory=list)
    backward_sub_steps: list[tuple[int, str]] = field(default_factory=list)
    permutation: list[int] | None = None
    blocks: list[list[int]] | None = None
    fill_in_count: int = 0
    annotated_expressions: list["AnnotatedLUExpression"] | None = None
    # Transposed solve steps: for solving (L*U)^T * x = b, i.e. U^T * L^T * x = b
    transpose_forward_sub_steps: list[tuple[int, str]] = field(default_factory=list)
    transpose_backward_sub_steps: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class AnnotatedLUExpression:
    """LU expression with species-dependency metadata for auto-reduction."""

    kind: str  # "L" or "U"
    row: int
    col: int
    expr: str
    depends_on: set[int]  # species indices whose activity affects this entry


@dataclass
class SparsityAnalysis:
    """Result of sparsity optimization pass."""

    original_nnz: int
    fill_in_positions: set[tuple[int, int]]
    total_nnz_after_fill: int
    permutation: list[int]
    inverse_permutation: list[int]
    blocks: list[list[int]]
    is_block_diagonal: bool
