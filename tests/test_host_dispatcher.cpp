#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <mkpp_host/dispatcher.hpp>
#include <mkpp_host/mechanism_manager.hpp>

using namespace mkpp::host;

struct DummyStandardSolver {
    bool integrate_called = false;
    KOKKOS_INLINE_FUNCTION void integrate(double dt, Kokkos::View<double*, Kokkos::LayoutLeft> sub_state, JvalsView<Kokkos::HostSpace> jvals) {
        integrate_called = true;
        (void)jvals;
        sub_state(0) += dt;
    }
};

struct DummyEnvSolver {
    bool integrate_called = false;
    double temp_used = 0.0;
    double rh_used = 0.0;
    KOKKOS_INLINE_FUNCTION void integrate(double dt, Kokkos::View<double*, Kokkos::LayoutLeft> sub_state, JvalsView<Kokkos::HostSpace> jvals, double temp, double rh) {
        integrate_called = true;
        (void)jvals;
        temp_used = temp;
        rh_used = rh;
        sub_state(0) += dt * temp;
    }
};

TEST(HostDispatcherTest, HostExecutionParamsDefaults) {
    HostExecutionParams params;
    EXPECT_DOUBLE_EQ(params.dt, 60.0);
    EXPECT_EQ(params.jvals.data(), nullptr);
    EXPECT_DOUBLE_EQ(params.temp, 288.15);
    EXPECT_DOUBLE_EQ(params.rh, 0.5);
    EXPECT_EQ(params.steps, 1);
}

TEST(HostDispatcherTest, TraitIntegrateCellStandard) {
    DummyStandardSolver solver;
    HostExecutionParams params;
    params.dt = 30.0;
    params.temp = 290.0;
    params.rh = 0.6;
    params.steps = 1;
    Kokkos::View<double*, Kokkos::LayoutLeft> state("state", 2);
    state(0) = 10.0;

    integrate_cell(solver, params, state);

    EXPECT_TRUE(solver.integrate_called);
    EXPECT_DOUBLE_EQ(state(0), 40.0);
}

TEST(HostDispatcherTest, TraitIntegrateCellEnv) {
    DummyEnvSolver solver;
    HostExecutionParams params;
    params.dt = 2.0;
    params.temp = 300.0;
    params.rh = 0.8;
    params.steps = 1;
    Kokkos::View<double*, Kokkos::LayoutLeft> state("state", 2);
    state(0) = 5.0;

    integrate_cell(solver, params, state);

    EXPECT_TRUE(solver.integrate_called);
    EXPECT_DOUBLE_EQ(solver.temp_used, 300.0);
    EXPECT_DOUBLE_EQ(solver.rh_used, 0.8);
    EXPECT_DOUBLE_EQ(state(0), 605.0);
}

TEST(HostDispatcherTest, MechanismRegistryMetadata) {
    auto mechs = MechanismRegistry::get_registered_mechanisms();
    EXPECT_GE(mechs.size(), 4u);

    EXPECT_TRUE(MechanismRegistry::has_mechanism("chapman"));
    EXPECT_TRUE(MechanismRegistry::has_mechanism("gocart"));
    EXPECT_TRUE(MechanismRegistry::has_mechanism("saprc99"));
    EXPECT_TRUE(MechanismRegistry::has_mechanism("ts1"));

    EXPECT_FALSE(MechanismRegistry::has_mechanism("invalid_mechanism"));

    auto chapman_info = MechanismRegistry::get_info("chapman");
    EXPECT_EQ(chapman_info.name, "chapman");
    EXPECT_GT(chapman_info.num_species, 0);

    auto gocart_info = MechanismRegistry::get_info("gocart");
    EXPECT_EQ(gocart_info.name, "gocart");
    EXPECT_TRUE(gocart_info.requires_env);
}

TEST(HostDispatcherTest, MechanismRegistryExecuteInvalidFailsFast) {
    Kokkos::View<double**, Kokkos::LayoutLeft> state("state", 10, 5);
    HostExecutionParams params;
    EXPECT_THROW(MechanismRegistry::execute("invalid_mech", state, params), std::runtime_error);
}

TEST(HostDispatcherTest, MechanismRegistryExecuteValid) {
    auto chapman_info = MechanismRegistry::get_info("chapman");
    Kokkos::View<double**, Kokkos::LayoutLeft> state("state", 10, chapman_info.num_species);

    auto host_state = Kokkos::create_mirror_view(state);
    for (int i = 0; i < 10; ++i) {
        for (int k = 0; k < chapman_info.num_species; ++k) {
            host_state(i, k) = 1.0e6;
        }
    }
    Kokkos::deep_copy(state, host_state);

    HostExecutionParams params;
    params.dt = 1.0;
    params.steps = 1;

    EXPECT_NO_THROW(MechanismRegistry::execute("chapman", state, params));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    return RUN_ALL_TESTS();
}
