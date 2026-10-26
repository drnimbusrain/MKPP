#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <string>
#include <vector>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#include TOSTRING(MECH_HEADER)

using ExecSpace = Kokkos::DefaultExecutionSpace;
using ConcentrationsView = Kokkos::View<double****, Kokkos::LayoutRight, typename ExecSpace::memory_space>;

#include "mkpp_host/dispatcher.hpp"

// ============================================================================
// KPP-compatible SUN computation (replicates Update_SUN from saprc99_diurnal)
// ============================================================================
static double compute_sun_kpp(double time_seconds) {
    const double PI = 3.14159265358979;
    const double SunRise = 4.5;  // hours
    const double SunSet = 19.5;  // hours

    double Thour = time_seconds / 3600.0;
    double Tlocal = Thour - static_cast<int>(Thour / 24) * 24;

    if (Tlocal >= SunRise && Tlocal <= SunSet) {
        double Ttmp = (2.0 * Tlocal - SunRise - SunSet) / (SunSet - SunRise);
        if (Ttmp > 0.0)
            Ttmp = Ttmp * Ttmp;
        else
            Ttmp = -Ttmp * Ttmp;
        return (1.0 + std::cos(PI * Ttmp)) / 2.0;
    }
    return 0.0;
}

// ============================================================================
// Compute SAPRC-99 photolysis J-values matching KPP's Update_RCONST formulas.
// Each jvals[i] = A_i * (optional_factor_i) * SUN / 60.0
// The A parameters and factors come directly from KPP's saprc99_diurnal_Rates.f90
// ============================================================================
static void compute_saprc99_jvals(double sun, double* jvals) {
    // jvals[i] corresponds to J_i in MKPP ordering (photo_idx from mechanism YAML)
    // KPP formula: RCONST(N) = A * (factor * SUN / 60.0)
    // For simple photolysis: rate = A * SUN/60
    // For compound: rate = A * factor * SUN/60

    const double s60 = sun / 60.0;  // SUN/60.0 factor used by KPP

    jvals[0] = 6.69e-1 * s60;               // J_0:  NO2 -> NO+O3P    (RCONST 1)
    jvals[1] = 1.59e0 * s60;                // J_1:  NO3 -> NO        (RCONST 15)
    jvals[2] = 1.50e+1 * s60;               // J_2:  NO3 -> NO2+O3P   (RCONST 16)
    jvals[3] = 3.76e-2 * s60;               // J_3:  O3 -> O3P        (RCONST 17)
    jvals[4] = 4.19e-3 * s60;               // J_4:  O3 -> O1D        (RCONST 18)
    jvals[5] = 1.27e-1 * s60;               // J_5:  HONO -> OH+NO    (RCONST 22)
    jvals[6] = 1.60e-2 * s60;               // J_6:  HONO -> HO2+NO2  (RCONST 23)
    jvals[7] = 5.40e-5 * s60;               // J_7:  HNO3 -> OH+NO2   (RCONST 28)
    jvals[8] = 4.69e-4 * s60;               // J_8:  HNO4 -> products  (RCONST 34)
    jvals[9] = 5.64e-4 * s60;               // J_9:  H2O2 -> OH       (RCONST 41)
    jvals[10] = 2.32e-3 * s60;              // J_10: HCHO -> HO2+CO   (RCONST 123)
    jvals[11] = 3.15e-3 * s60;              // J_11: HCHO -> CO       (RCONST 124)
    jvals[12] = 4.16e-4 * s60;              // J_12: CCHO -> products  (RCONST 131)
    jvals[13] = 1.40e-3 * s60;              // J_13: RCHO -> products  (RCONST 134)
    jvals[14] = 4.16e-5 * s60;              // J_14: ACET -> products  (RCONST 137)
    jvals[15] = 9.49e-4 * (1.50e-1 * s60);  // J_15: MEK -> products   (RCONST 139)
    jvals[16] = 3.94e-4 * s60;              // J_16: COOH -> products  (RCONST 142)
    jvals[17] = 3.94e-4 * s60;              // J_17: ROOH -> products  (RCONST 144)
    jvals[18] = 8.93e-3 * s60;              // J_18: GLY -> CO+HO2    (RCONST 145)
    jvals[19] = 1.81e-1 * (6.00e-3 * s60);  // J_19: GLY -> HCHO+CO   (RCONST 146)
    jvals[20] = 1.10e-2 * s60;              // J_20: MGLY -> products  (RCONST 149)
    jvals[21] = 1.90e-2 * s60;              // J_21: BACL -> products  (RCONST 152)
    jvals[22] = 6.22e-2 * (5.00e-2 * s60);  // J_22: BALD -> XC       (RCONST 159)
    jvals[23] = 3.32e-2 * (4.10e-3 * s60);  // J_23: METHACRO         (RCONST 165)
    jvals[24] = 3.32e-2 * (2.10e-3 * s60);  // J_24: MVK              (RCONST 169)
    jvals[25] = 3.32e-2 * (4.10e-3 * s60);  // J_25: ISOPROD          (RCONST 173)
    jvals[26] = 9.49e-4 * (2.00e-2 * s60);  // J_26: PROD2            (RCONST 175)
    jvals[27] = 2.35e-4 * s60;              // J_27: RNO3             (RCONST 177)
    jvals[28] = 2.06e-1 * (3.65e-1 * s60);  // J_28: DCB2             (RCONST 181)
    jvals[29] = 3.32e-2 * (7.28e0 * s60);   // J_29: DCB3             (RCONST 183)
}

TEST(E2ESolverValidation, MechanismIntegration) {
    int cells = 1;
    if (const char* env_c = std::getenv("NUM_CELLS")) { cells = std::atoi(env_c); }
    int steps = 1;
    if (const char* env_s = std::getenv("NUM_STEPS")) { steps = std::atoi(env_s); }

    int n_spec = NUM_SPECIES_MACRO;
    if (n_spec == 0) n_spec = 100;

    std::vector<double> host_data(cells * n_spec * 1 * 1, 1.0e8);

    // Match initial concentrations from KPP def using exact MKPP species indices
    const double cfactor = 2.447600e+13;
    if (n_spec >= 79) {
        for (int i = 0; i < cells; ++i) {
            double* c = &host_data[i * n_spec];
            for (int k = 0; k < n_spec; ++k) c[k] = 0.0;
            c[2] = (0.1) * cfactor;         // NO
            c[3] = (0.05) * cfactor;        // NO2
            c[6] = (0.001) * cfactor;       // HONO
            c[9] = (0.05) * cfactor;        // SO2
            c[12] = (0.01121) * cfactor;    // HCHO
            c[13] = (0.002316) * cfactor;   // CCHO
            c[14] = (0.00172) * cfactor;    // RCHO
            c[15] = (0.00507) * cfactor;    // ACET
            c[16] = (0.00326) * cfactor;    // MEK
            c[17] = (0.000677) * cfactor;   // HCOOH
            c[18] = (0.00589) * cfactor;    // MEOH
            c[19] = (0.00116) * cfactor;    // CCO_OH
            c[20] = (0.000392) * cfactor;   // RCO_OH
            c[21] = (0.000121) * cfactor;   // GLY
            c[22] = (8.37e-05) * cfactor;   // MGLY
            c[24] = (0.00056) * cfactor;    // CRES
            c[25] = (7.51e-05) * cfactor;   // BALD
            c[26] = (8.93e-05) * cfactor;   // ISOPROD
            c[27] = (0.0013) * cfactor;     // METHACRO
            c[29] = (0.00193) * cfactor;    // PROD2
            c[33] = (0.0189) * cfactor;     // ETHENE
            c[34] = (0.000433) * cfactor;   // ISOPRENE
            c[35] = (0.01167) * cfactor;    // ALK1
            c[36] = (0.0188) * cfactor;     // ALK2
            c[37] = (0.0469) * cfactor;     // ALK3
            c[38] = (0.0417) * cfactor;     // ALK4
            c[39] = (0.0306) * cfactor;     // ALK5
            c[40] = (0.0118) * cfactor;     // ARO1
            c[41] = (0.00874) * cfactor;    // ARO2
            c[42] = (0.0104) * cfactor;     // OLE1
            c[43] = (0.00797) * cfactor;    // OLE2
            c[44] = (0.00082) * cfactor;    // TERP
            c[47] = (0.000606) * cfactor;   // PHEN
            c[56] = (0.2) * cfactor;        // XC
            c[57] = (7.843e-09) * cfactor;  // O3P
            c[74] = (1000000.0) * cfactor;  // AIR
            c[75] = (209000.0) * cfactor;   // O2
            c[76] = (20000.0) * cfactor;    // H2O
            c[78] = (1.0) * cfactor;        // CH4
        }
    }

    double dt_step = 60.0;
    if (const char* env_dt = std::getenv("DT_STEP")) { dt_step = std::atof(env_dt); }

    ConcentrationsView state(host_data.data(), cells, n_spec, 1, 1);
    const bool use_serial_host = []() {
        const char* mode = std::getenv("MKPP_EXECUTION_MODE");
        return mode != nullptr && std::string(mode) == "serial";
    }();

    // Photolysis J-values: computed per-step to match KPP's diurnal SUN cycle.
    // For mechanisms without photolysis (n_spec < 79), jvals remains zero.
    std::vector<double> jvals_data(64, 0.0);
    mkpp::host::JvalsView<typename ExecSpace::memory_space> jvals(jvals_data.data(), jvals_data.size());

    auto start = std::chrono::high_resolution_clock::now();

    if (n_spec >= 79) {
        // SAPRC-99 mode: KPP's Update_RCONST hardcodes SUN = 1.0_dp (constant daytime)
        // The diurnal Update_SUN subroutine is NOT called in KPP's main program.
        MECH_SOLVER_NAMESPACE::SolverKernels<ExecSpace> solver;
        auto sub_state = Kokkos::subview(state, 0, Kokkos::ALL(), 0, 0);

        // Compute jvals once with SUN=1.0 (constant, matching KPP's Update_RCONST)
        compute_saprc99_jvals(1.0, jvals_data.data());

        for (int step = 0; step < steps; ++step) { solver.integrate(dt_step, sub_state, jvals); }
    } else {
        // Non-SAPRC mechanisms: use fixed zero jvals (no photolysis)
        if (use_serial_host) {
            mkpp::host::execute_mechanism_serial_steps<MECH_SOLVER_NAMESPACE::SolverKernels<ExecSpace>>(
                state, dt_step, steps, jvals);
        } else {
            mkpp::host::execute_mechanism_steps<MECH_SOLVER_NAMESPACE::SolverKernels<ExecSpace>>(
                TOSTRING(MECH_HEADER), state, dt_step, steps, jvals);
        }
    }

    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Execution mode: " << (use_serial_host ? "serial" : "kokkos") << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsed.count() << " ms\n";

    std::ofstream out("e2e_output.csv");
    out << "time_step,species_name,concentration\n";
    for (int k = 0; k < n_spec; k++) {
        out << steps << "," << k << "," << std::scientific << std::setprecision(12) << host_data[k]
            << "\n";
    }
    out.close();
}

int main(int argc, char** argv) {
    // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
