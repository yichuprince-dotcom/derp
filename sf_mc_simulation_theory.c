#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <limits.h>
#include <errno.h>


#ifndef GAMMA_VALUES
#define GAMMA_VALUES 2.2, 2.5, 2.8
#endif
#ifndef R_VALUES
#define R_VALUES 2, 3, 4
#endif
const double GAMMA_LIST[] = {GAMMA_VALUES};
const int NUM_GAMMA = sizeof(GAMMA_LIST) / sizeof(GAMMA_LIST[0]);
const int R_LIST[] = {R_VALUES};
const int NUM_R = sizeof(R_LIST) / sizeof(R_LIST[0]);

#ifndef N
#define N 20000
#endif
#ifndef K_MIN
#define K_MIN 1
#endif
#ifndef K_MAX
#define K_MAX 20000
#endif
#ifndef MAX_R_CONFIG
#define MAX_R_CONFIG 4
#endif
#ifndef N_RUN_MAX
#define N_RUN_MAX 100
#endif
#ifndef DEGREE_CUTOFF
#define DEGREE_CUTOFF 0
#endif
#ifndef TOLERANCE
#define TOLERANCE 1e-10
#endif
#ifndef MAX_ITER
#define MAX_ITER 100000
#endif
#ifndef RANDOM_SEED
#define RANDOM_SEED 0
#endif
#ifndef GRAPH_RESTARTS
#define GRAPH_RESTARTS 20
#endif
#ifndef REPAIR_TRIES
#define REPAIR_TRIES 5000
#endif
#ifndef SWITCHES_PER_EDGE
#define SWITCHES_PER_EDGE 5
#endif
#define P_STEPS 100


typedef struct {
    int* neighbors;
    int capacity;
    int count;
} AdjNode;

typedef struct {
    AdjNode* nodes;
    int node_count;
    long long E;
} Graph;

typedef struct {
    long long s, u, gin_t, gin_u, gout_t, gout_u;
} ComponentCounts;

int **out_knn, **in_knn;
int *k_out, *k_in, *dam1;
double P_k[K_MAX + 1];
double theory_avg_k;
double theory_p;
double W_plus[MAX_R_CONFIG + 1], W_minus[MAX_R_CONFIG + 1];
double S_plus, S_minus, U_plus, U_minus;
double R_trusted, R_untrusted;
double CURRENT_GAMMA;
int CURRENT_R;
int CURRENT_K_CUTOFF;
int theory_iterations;
double theory_residual;
static uint64_t rng_state;
static uint64_t master_seed;


int run_single_config(void);
void* checked_calloc(size_t count, size_t size);
uint64_t random_u64(void);
size_t random_index(size_t bound);
double uniform_random(void);
void shuffle(int* array, size_t n);
int generate_powerlaw_sequence(int* sequence, int num_nodes, double gamma,
                              int k_min, int cutoff);
Graph* create_graph_from_sequences(const int* in_degree, const int* out_degree,
                                   int num_nodes, long long E);
void free_graph(Graph* g);
void install_graph(const Graph* g);
void release_adjacency(void);
void Constrained_DFS(int u, int rm, int** adj, int* k, int* visited, int* drange);
void calculate_components(ComponentCounts* counts);
void initialize_theory(const int* degree_seq, int num_nodes);
int solve_W_equations_theory(void);
void calculate_S_U_theory(void);
void calculate_R_theory(void);
int perform_theoretical_run(void);
int save_pk_comparison(const int* degree_seq, int num_nodes, const char* filename);
static double node_response(double theta, int k);
static double edge_response(double theta);
static int write_run_metadata(uint64_t config_state, long long edges);


int main(int argc, char** argv) {
    if (N < 3 || N_RUN_MAX < 1 || K_MIN < 1 || K_MIN > K_MAX ||
        TOLERANCE <= 0 || MAX_ITER < 1 || GRAPH_RESTARTS < 1 || REPAIR_TRIES < 1 ||
        SWITCHES_PER_EDGE < 0 || DEGREE_CUTOFF < 0) {
        fprintf(stderr, "Error: Invalid parameter configuration.\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < NUM_GAMMA; ++i) {
        if (!isfinite(GAMMA_LIST[i]) || GAMMA_LIST[i] <= 1.0) {
            fprintf(stderr, "Error: Every GAMMA value must be finite and > 1.\n");
            return EXIT_FAILURE;
        }
    }
    for (int i = 0; i < NUM_R; ++i) {
        if (R_LIST[i] < 1 || R_LIST[i] > MAX_R_CONFIG) {
            fprintf(stderr, "Error: Every R must be in [1, MAX_R_CONFIG].\n");
            return EXIT_FAILURE;
        }
    }
    CURRENT_K_CUTOFF = DEGREE_CUTOFF > 0 ? DEGREE_CUTOFF : (int)sqrt((double)N);
    if (CURRENT_K_CUTOFF > K_MAX) CURRENT_K_CUTOFF = K_MAX;
    if (CURRENT_K_CUTOFF > (N - 1) / 2) CURRENT_K_CUTOFF = (N - 1) / 2;
    if (CURRENT_K_CUTOFF < K_MIN) {
        fprintf(stderr, "Error: No admissible degree range for this N/K_MIN.\n");
        return EXIT_FAILURE;
    }
    master_seed = RANDOM_SEED ? (uint64_t)RANDOM_SEED : (uint64_t)time(NULL);
    if (argc > 2) {
        fprintf(stderr, "Usage: %s [unsigned_decimal_seed]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 2) {
        char* end;
        errno = 0;
        if (argv[1][0] < '0' || argv[1][0] > '9') return EXIT_FAILURE;
        master_seed = (uint64_t)strtoull(argv[1], &end, 10);
        if (errno || *end) {
            fprintf(stderr, "Error: Invalid random seed.\n");
            return EXIT_FAILURE;
        }
    }

    rng_state = master_seed ? master_seed : UINT64_C(0x9e3779b97f4a7c15);
    printf("Master seed = %" PRIu64 "; actual degree cutoff = %d\n", master_seed, CURRENT_K_CUTOFF);

    for (int i = 0; i < NUM_GAMMA; ++i) {
        for (int j = 0; j < NUM_R; ++j) {
            CURRENT_GAMMA = GAMMA_LIST[i];
            CURRENT_R = R_LIST[j];
            printf("\n###########################################################\n");
            printf("### SF POS: GAMMA = %.2f, R = %d, N = %d ###\n", CURRENT_GAMMA, CURRENT_R, N);
            printf("###########################################################\n\n");
            if (!run_single_config()) return EXIT_FAILURE;
        }
    }
    printf("All parameter combinations have been executed.\n");
    return EXIT_SUCCESS;
}

int run_single_config(void) {
    uint64_t config_state = rng_state;
    printf(">> Step 1: Generate one shared in/out degree sequence\n");
    int* degree_sf = checked_calloc(N, sizeof(int));
    if (!generate_powerlaw_sequence(degree_sf, N, CURRENT_GAMMA, K_MIN, CURRENT_K_CUTOFF)) {
        free(degree_sf);
        return 0;
    }
    long long edges = 0;
    for (int i = 0; i < N; ++i) edges += degree_sf[i];
    printf("   Target edges = %lld, <k> = %.6f, kin(i) = kout(i)\n", edges, (double)edges / N);

    char filename[160];
    snprintf(filename, sizeof(filename), "comparison_pk_SF_Pos_G%.2f_R%d_kmin%d.txt",
             CURRENT_GAMMA, CURRENT_R, K_MIN);
    if (!write_run_metadata(config_state, edges) || !save_pk_comparison(degree_sf, N, filename)) {
        free(degree_sf);
        return 0;
    }

    printf("\n>> Step 2: Theory using the same empirical P(k)\n");
    initialize_theory(degree_sf, N);
    if (!perform_theoretical_run()) {
        free(degree_sf);
        return 0;
    }

    printf("\n>> Step 3: Simulation (%d realizations)\n", N_RUN_MAX);
    ComponentCounts totals[P_STEPS + 1] = {{0}};
    double* trust_draws = checked_calloc(N, sizeof(double));
    dam1 = checked_calloc(N, sizeof(int));
    for (int nrun = 0; nrun < N_RUN_MAX; ++nrun) {

        Graph* g = create_graph_from_sequences(degree_sf, degree_sf, N, edges);
        if (!g) {
            fprintf(stderr, "Error: Could not realize the degree sequence without illegal edges.\n"
                    "No realization was silently dropped; lower DEGREE_CUTOFF if necessary.\n");
            free(degree_sf); free(trust_draws); free(dam1); dam1 = NULL;
            return 0;
        }
        install_graph(g);
        for (int i = 0; i < N; ++i) trust_draws[i] = uniform_random();
        for (int nc = 0; nc <= P_STEPS; ++nc) {
            double p = (double)(P_STEPS - nc) / P_STEPS;
            for (int i = 0; i < N; ++i) dam1[i] = trust_draws[i] < p;
            ComponentCounts one;
            calculate_components(&one);
            totals[nc].s += one.s; totals[nc].u += one.u;
            totals[nc].gin_t += one.gin_t; totals[nc].gin_u += one.gin_u;
            totals[nc].gout_t += one.gout_t; totals[nc].gout_u += one.gout_u;
        }
        release_adjacency();
        free_graph(g);
        printf("   -> Completed simulation %d/%d\n", nrun + 1, N_RUN_MAX);
    }

    printf("\n>> Step 4: Save simulation results\n");
    snprintf(filename, sizeof(filename), "SF_Pos_SIM_G%.2f_R%d_kmin%d.txt",
             CURRENT_GAMMA, CURRENT_R, K_MIN);
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror(filename);
        free(degree_sf); free(trust_draws); free(dam1); dam1 = NULL;
        return 0;
    }
    fprintf(fp, "# p_trusted\tEGSCC_T\tEGSCC_UT\tGIN_T\tGIN_UT\tGOUT_T\tGOUT_UT\n");
    double denominator = (double)N_RUN_MAX * N;
    for (int nc = 0; nc <= P_STEPS; ++nc) {
        fprintf(fp, "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                (double)(P_STEPS - nc) / P_STEPS,
                totals[nc].s / denominator, totals[nc].u / denominator,
                totals[nc].gin_t / denominator, totals[nc].gin_u / denominator,
                totals[nc].gout_t / denominator, totals[nc].gout_u / denominator);
    }
    int write_failed = ferror(fp);
    if (fclose(fp)) write_failed = 1;
    free(degree_sf); free(trust_draws); free(dam1); dam1 = NULL;
    if (write_failed) { fprintf(stderr, "Error: Failed to write %s\n", filename); return 0; }
    printf("   Saved to %s\n", filename);
    return 1;
}


static double node_response(double theta, int k) {
    if (k == 0 || theta <= 0.0) return 0.0;
    if (theta >= 1.0) return 1.0;
    return -expm1((double)k * log1p(-theta));
}

static double edge_response(double theta) {
    double result = 0.0;
    for (int k = 1; k <= K_MAX; ++k) {
        if (P_k[k] > 0.0) result += (double)k * P_k[k] * node_response(theta, k);
    }
    return result / theory_avg_k;
}

void initialize_theory(const int* degree_seq, int num_nodes) {
    memset(P_k, 0, sizeof(P_k));
    theory_avg_k = 0.0;
    for (int i = 0; i < num_nodes; ++i) {
        int k = degree_seq[i];
        if (k < 0 || k > K_MAX) { fprintf(stderr, "Error: Degree outside theoretical support.\n"); exit(EXIT_FAILURE); }
        P_k[k] += 1.0 / num_nodes;
        theory_avg_k += (double)k / num_nodes;
    }
}

int solve_W_equations_theory(void) {
    memset(W_plus, 0, sizeof(W_plus)); memset(W_minus, 0, sizeof(W_minus));
    theory_iterations = 0; theory_residual = 0.0;
    if (theory_p <= 0.0 || theory_avg_k == 0.0) return 1;


    double theta[MAX_R_CONFIG + 1] = {0};
    double next[MAX_R_CONFIG + 1] = {0};
    for (int r = 1; r <= CURRENT_R; ++r) theta[r] = 1.0;
    int converged = 0;
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double trusted_term = theory_p * edge_response(theta[CURRENT_R]);
        theory_residual = 0.0;
        for (int r = 1; r <= CURRENT_R; ++r) {
            next[r] = trusted_term + (1.0 - theory_p) * edge_response(theta[r - 1]);
            if (!isfinite(next[r])) return 0;
            if (next[r] < 0.0) next[r] = 0.0;
            if (next[r] > 1.0) next[r] = 1.0;
            double change = fabs(next[r] - theta[r]);
            if (change > theory_residual) theory_residual = change;
        }
        memcpy(theta, next, sizeof(theta));
        theory_iterations = iter + 1;
        if (theory_residual < TOLERANCE) { converged = 1; break; }
    }
    for (int r = 1; r <= CURRENT_R; ++r) W_plus[r] = W_minus[r] = theta[r] - theta[r - 1];
    return converged;
}

void calculate_S_U_theory(void) {
    double all = 0.0, relays = 0.0;
    for (int r = 1; r <= CURRENT_R; ++r) {
        all += W_plus[r];
        if (r < CURRENT_R) relays += W_plus[r];
    }
    double s = 0.0, u = 0.0;
    for (int k = 0; k <= K_MAX; ++k) {
        if (P_k[k] == 0.0) continue;
        s += P_k[k] * node_response(all, k);
        u += P_k[k] * node_response(relays, k);
    }
    S_plus = S_minus = theory_p * s;
    U_plus = U_minus = (1.0 - theory_p) * u;
}

void calculate_R_theory(void) {
    double all = 0.0, relays = 0.0;
    for (int r = 1; r <= CURRENT_R; ++r) {
        all += W_plus[r];
        if (r < CURRENT_R) relays += W_plus[r];
    }
    double s = 0.0, u = 0.0;


    for (int k = 0; k <= K_MAX; ++k) {
        if (P_k[k] == 0.0) continue;
        double a = node_response(all, k), b = node_response(relays, k);
        s += P_k[k] * a * a;
        u += P_k[k] * b * b;
    }
    R_trusted = theory_p * s;
    R_untrusted = (1.0 - theory_p) * u;
}

int perform_theoretical_run(void) {
    char filename[160], diagnostic[160];
    snprintf(filename, sizeof(filename), "SF_Pos_THEORY_G%.2f_R%d_kmin%d.txt", CURRENT_GAMMA, CURRENT_R, K_MIN);
    snprintf(diagnostic, sizeof(diagnostic), "SF_Pos_SOLVER_G%.2f_R%d_kmin%d.txt", CURRENT_GAMMA, CURRENT_R, K_MIN);
    FILE* fp = fopen(filename, "w");
    FILE* log = fopen(diagnostic, "w");
    if (!fp || !log) { perror("Theory output"); if (fp) fclose(fp); if (log) fclose(log); return 0; }
    fprintf(fp, "# p_trusted\tS_plus\tS_minus\tU_plus\tU_minus\tSCGC_T\tSCGC_UT\n");
    fprintf(log, "# p_trusted\titerations\tmax_update\tconverged\n");
    int ok = 1;
    for (int i = P_STEPS; i >= 0; --i) {
        theory_p = (double)i / P_STEPS;
        int converged = solve_W_equations_theory();
        fprintf(log, "%.6f\t%d\t%.12e\t%d\n", theory_p, theory_iterations, theory_residual, converged);
        if (!converged) {
            fprintf(stderr, "Error: Theory did not converge at p=%.6f; output is incomplete.\n", theory_p);
            ok = 0; break;
        }
        calculate_S_U_theory(); calculate_R_theory();
        fprintf(fp, "%.4f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                theory_p, S_plus, S_minus, U_plus, U_minus, R_trusted, R_untrusted);
    }
    if (ferror(fp) || ferror(log)) ok = 0;
    if (fclose(fp)) ok = 0;
    if (fclose(log)) ok = 0;
    return ok;
}

int save_pk_comparison(const int* degree_seq, int num_nodes, const char* filename) {
    int* counts = checked_calloc(K_MAX + 1, sizeof(int));
    for (int i = 0; i < num_nodes; ++i) ++counts[degree_seq[i]];
    double normalization = 0.0;
    for (int k = K_MIN; k <= CURRENT_K_CUTOFF; ++k) normalization += pow((double)k / K_MIN, -CURRENT_GAMMA);
    FILE* fp = fopen(filename, "w");
    if (!fp) { perror(filename); free(counts); return 0; }
    fprintf(fp, "# k\tTheoretical_P(k)\tEmpirical_P(k)\n");
    for (int k = 0; k <= CURRENT_K_CUTOFF; ++k) {
        double expected = k >= K_MIN ? pow((double)k / K_MIN, -CURRENT_GAMMA) / normalization : 0.0;
        fprintf(fp, "%d\t%.12e\t%.12e\n", k, expected, (double)counts[k] / num_nodes);
    }
    free(counts);
    int ok = !ferror(fp);
    if (fclose(fp)) ok = 0;
    return ok;
}


void Constrained_DFS(int u, int rm, int** adj, int* k, int* visited, int* drange) {


    if (visited[u] && drange[u] <= rm) return;
    int* queue = checked_calloc(N, sizeof(int));
    unsigned char* queued = checked_calloc(N, sizeof(unsigned char));
    size_t head = 0, tail = 0, count = 1;
    queue[tail] = u; tail = (tail + 1) % N;
    queued[u] = 1; visited[u] = 1; drange[u] = rm;
    while (count) {
        int a = queue[head]; head = (head + 1) % N; --count; queued[a] = 0;
        for (int j = 0; j < k[a]; ++j) {
            int b = adj[a][j];
            int distance = dam1[b] ? 1 : drange[a] + 1;
            if (distance > CURRENT_R || distance >= drange[b]) continue;
            drange[b] = distance; visited[b] = 1;
            if (!queued[b]) {
                queue[tail] = b; tail = (tail + 1) % N; ++count; queued[b] = 1;
            }
        }
    }
    free(queue); free(queued);
}

void calculate_components(ComponentCounts* counts) {
    memset(counts, 0, sizeof(*counts));
    int* component_map = checked_calloc(N, sizeof(int));
    int* forward = checked_calloc(N, sizeof(int));
    int* backward = checked_calloc(N, sizeof(int));
    int* distance = checked_calloc(N, sizeof(int));
    long long largest_trusted = 0;
    for (int seed = 0; seed < N; ++seed) {
        if (!dam1[seed] || component_map[seed]) continue;
        memset(forward, 0, N * sizeof(int));
        memset(backward, 0, N * sizeof(int));
        for (int i = 0; i < N; ++i) distance[i] = CURRENT_R + 1;
        Constrained_DFS(seed, 1, out_knn, k_out, forward, distance);
        for (int i = 0; i < N; ++i) distance[i] = CURRENT_R + 1;
        Constrained_DFS(seed, 1, in_knn, k_in, backward, distance);
        long long trusted_count = 0;
        for (int i = 0; i < N; ++i) {
            if (dam1[i] && forward[i] && backward[i]) { component_map[i] = 1; ++trusted_count; }
        }
        if (trusted_count <= largest_trusted) continue;
        largest_trusted = trusted_count;
        memset(counts, 0, sizeof(*counts));


        for (int i = 0; i < N; ++i) {
            if (forward[i] && backward[i]) { if (dam1[i]) ++counts->s; else ++counts->u; }
            if (backward[i]) { if (dam1[i]) ++counts->gin_t; else ++counts->gin_u; }
            if (forward[i]) { if (dam1[i]) ++counts->gout_t; else ++counts->gout_u; }
        }
    }
    free(component_map); free(forward); free(backward); free(distance);
}

int generate_powerlaw_sequence(int* sequence, int num_nodes, double gamma, int k_min, int cutoff) {
    if (num_nodes < 1 || k_min < 1 || cutoff < k_min || !isfinite(gamma) || gamma <= 1.0) return 0;
    double* cdf = checked_calloc((size_t)cutoff + 1, sizeof(double));
    double total = 0.0;


    for (int k = k_min; k <= cutoff; ++k) { total += pow((double)k / k_min, -gamma); cdf[k] = total; }
    for (int k = k_min; k <= cutoff; ++k) cdf[k] /= total;
    cdf[cutoff] = 1.0;
    for (int i = 0; i < num_nodes; ++i) {
        double u = uniform_random();
        int lo = k_min, hi = cutoff;
        while (lo < hi) { int mid = lo + (hi - lo) / 2; if (u < cdf[mid]) hi = mid; else lo = mid + 1; }
        sequence[i] = lo;
    }
    free(cdf);
    return 1;
}

static int edge_exists_except(const Graph* g, int from, int to, int u, int j, int a, int b) {
    for (int i = 0; i < g->nodes[from].count; ++i) {
        if ((from == u && i == j) || (from == a && i == b)) continue;
        if (g->nodes[from].neighbors[i] == to) return 1;
    }
    return 0;
}

static int edge_is_valid(const Graph* g, int u, int j) {
    int v = g->nodes[u].neighbors[j];
    return u != v && !edge_exists_except(g, u, v, u, j, -1, -1)
                  && !edge_exists_except(g, v, u, u, j, -1, -1);
}

static int try_switch(Graph* g, int u, int j, int a, int b) {
    int v = g->nodes[u].neighbors[j], w = g->nodes[a].neighbors[b];

    if (u == a || v == w || u == w || a == v || (u == v && a == w)) return 0;
    if (edge_exists_except(g, u, w, u, j, a, b) || edge_exists_except(g, w, u, u, j, a, b) ||
        edge_exists_except(g, a, v, u, j, a, b) || edge_exists_except(g, v, a, u, j, a, b)) return 0;
    g->nodes[u].neighbors[j] = w;
    g->nodes[a].neighbors[b] = v;
    return 1;
}

Graph* create_graph_from_sequences(const int* in_degree, const int* out_degree, int num_nodes, long long E) {
    if (num_nodes < 3 || E <= 0 || (uint64_t)E > SIZE_MAX / sizeof(int)) return NULL;
    long long sum_in = 0, sum_out = 0;
    for (int i = 0; i < num_nodes; ++i) {
        if (in_degree[i] < 0 || in_degree[i] != out_degree[i] || in_degree[i] > (num_nodes - 1) / 2) return NULL;
        sum_in += in_degree[i]; sum_out += out_degree[i];
    }
    if (sum_in != E || sum_out != E) return NULL;
    size_t m = (size_t)E;
    int* destinations = checked_calloc(m, sizeof(int));
    int* edge_source = checked_calloc(m, sizeof(int));
    int* edge_slot = checked_calloc(m, sizeof(int));
    Graph* g = checked_calloc(1, sizeof(Graph));
    g->node_count = num_nodes; g->E = E;
    g->nodes = checked_calloc((size_t)num_nodes, sizeof(AdjNode));
    size_t index = 0;
    for (int i = 0; i < num_nodes; ++i) {
        g->nodes[i].count = g->nodes[i].capacity = out_degree[i];
        g->nodes[i].neighbors = checked_calloc((size_t)out_degree[i], sizeof(int));
        for (int j = 0; j < out_degree[i]; ++j) { edge_source[index] = i; edge_slot[index] = j; destinations[index++] = i; }
    }
    int success = 0;
    for (int attempt = 0; attempt < GRAPH_RESTARTS && !success; ++attempt) {
        shuffle(destinations, m);
        for (size_t e = 0; e < m; ++e) g->nodes[edge_source[e]].neighbors[edge_slot[e]] = destinations[e];
        success = 1;
        for (size_t e = 0; e < m; ++e) {
            int u = edge_source[e], j = edge_slot[e];
            if (edge_is_valid(g, u, j)) continue;
            int repaired = 0;
            for (int trial = 0; trial < REPAIR_TRIES && !repaired; ++trial) {
                size_t other = random_index(m);
                repaired = try_switch(g, u, j, edge_source[other], edge_slot[other]);
            }
            if (!repaired) { success = 0; break; }
        }
    }
    if (success) {


        for (int sweep = 0; sweep < SWITCHES_PER_EDGE; ++sweep) {
            for (size_t e = 0; e < m; ++e) {
                size_t a = random_index(m), b = random_index(m);
                try_switch(g, edge_source[a], edge_slot[a], edge_source[b], edge_slot[b]);
            }
        }
        int* realized_in = checked_calloc((size_t)num_nodes, sizeof(int));
        for (size_t e = 0; e < m; ++e) {
            int u = edge_source[e], j = edge_slot[e];
            if (!edge_is_valid(g, u, j)) success = 0;
            ++realized_in[g->nodes[u].neighbors[j]];
        }
        for (int i = 0; i < num_nodes; ++i) if (realized_in[i] != in_degree[i]) success = 0;
        free(realized_in);
    }
    free(destinations); free(edge_source); free(edge_slot);
    if (!success) { free_graph(g); return NULL; }
    return g;
}

void install_graph(const Graph* g) {
    k_out = checked_calloc(N, sizeof(int)); k_in = checked_calloc(N, sizeof(int));
    out_knn = checked_calloc(N, sizeof(int*)); in_knn = checked_calloc(N, sizeof(int*));
    for (int i = 0; i < N; ++i) {
        k_out[i] = g->nodes[i].count;
        out_knn[i] = g->nodes[i].neighbors;
        for (int j = 0; j < k_out[i]; ++j) ++k_in[out_knn[i][j]];
    }
    for (int i = 0; i < N; ++i) {
        if (k_in[i] != k_out[i]) { fprintf(stderr, "Error: MC degree equality violated.\n"); exit(EXIT_FAILURE); }
        in_knn[i] = checked_calloc((size_t)k_in[i], sizeof(int));
    }
    int* filled = checked_calloc(N, sizeof(int));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < k_out[i]; ++j) { int v = out_knn[i][j]; in_knn[v][filled[v]++] = i; }
    }
    free(filled);
}

void release_adjacency(void) {
    for (int i = 0; i < N; ++i) free(in_knn[i]);
    free(in_knn); free(out_knn); free(k_in); free(k_out);
    in_knn = out_knn = NULL; k_in = k_out = NULL;
}

void free_graph(Graph* g) {
    if (!g) return;
    for (int i = 0; i < g->node_count; ++i) free(g->nodes[i].neighbors);
    free(g->nodes); free(g);
}


void* checked_calloc(size_t count, size_t size) {
    if (size && count > SIZE_MAX / size) { fprintf(stderr, "Error: Allocation size overflow.\n"); exit(EXIT_FAILURE); }
    void* pointer = calloc(count ? count : 1, size);
    if (!pointer) { fprintf(stderr, "Error: Memory allocation failed.\n"); exit(EXIT_FAILURE); }
    return pointer;
}

uint64_t random_u64(void) {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * UINT64_C(2685821657736338717);
}

size_t random_index(size_t bound) {
    if (!bound) { fprintf(stderr, "Error: Empty random-index range.\n"); exit(EXIT_FAILURE); }
    uint64_t b = (uint64_t)bound, threshold = (UINT64_C(0) - b) % b, value;
    do { value = random_u64(); } while (value < threshold);
    return (size_t)(value % b);
}

double uniform_random(void) { return (double)(random_u64() >> 11) * 0x1.0p-53; }

void shuffle(int* array, size_t n) {
    for (size_t i = n; i > 1; --i) {
        size_t j = random_index(i);
        int temp = array[i - 1]; array[i - 1] = array[j]; array[j] = temp;
    }
}

static int write_run_metadata(uint64_t config_state, long long edges) {
    char filename[160];
    snprintf(filename, sizeof(filename), "SF_Pos_CONFIG_G%.2f_R%d_kmin%d.txt", CURRENT_GAMMA, CURRENT_R, K_MIN);
    FILE* fp = fopen(filename, "w");
    if (!fp) { perror(filename); return 0; }
    fprintf(fp, "# Newly reconstructed MC-SF DERP implementation\n");
    fprintf(fp, "N=%d\nN_RUN_MAX=%d\nGAMMA=%.17g\nR=%d\nK_MIN=%d\nK_MAX=%d\nACTUAL_CUTOFF=%d\n",
            N, N_RUN_MAX, CURRENT_GAMMA, CURRENT_R, K_MIN, K_MAX, CURRENT_K_CUTOFF);
    fprintf(fp, "MASTER_SEED=%" PRIu64 "\nCONFIG_RNG_STATE=%" PRIu64 "\nEDGES=%lld\n", master_seed, config_state, edges);
    fprintf(fp, "TOLERANCE=%.17g\nMAX_ITER=%d\nGRAPH_RESTARTS=%d\nREPAIR_TRIES=%d\nSWITCHES_PER_EDGE=%d\n",
            (double)TOLERANCE, MAX_ITER, GRAPH_RESTARTS, REPAIR_TRIES, SWITCHES_PER_EDGE);
    fprintf(fp, "DEGREE_MODEL=discrete truncated power law; one degree sequence per gamma/R\n");
    fprintf(fp, "GRAPH_MODEL=exact kin=kout; no self/duplicate/reciprocal edges; finite swap mixing\n");
    fprintf(fp, "THEORY_MODEL=locally tree-like ensemble approximation using empirical P(k)\n");
    fprintf(fp, "NORMALIZATION=all fractions divided by total N, not trusted population\n");
    fprintf(fp, "FINITE_COMPONENT_RULE=largest SCC by trusted-node count; first-seed tie\n");
    fprintf(fp, "STATUS=configuration only; success requires normal program exit and complete outputs\n");
    int ok = !ferror(fp);
    if (fclose(fp)) ok = 0;
    return ok;
}
