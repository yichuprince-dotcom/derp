#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


const double GAMMA_LIST[] = {2.2, 2.5, 2.8};
const int NUM_GAMMA = sizeof(GAMMA_LIST) / sizeof(GAMMA_LIST[0]);

const int R_LIST[] = {2, 3, 4};
const int NUM_R = sizeof(R_LIST) / sizeof(R_LIST[0]);


#define N 20000
#define K_MIN 1
#define K_MAX 20000
#define MAX_R_CONFIG 4


#define N_RUN_MAX 100


#define TOLERANCE 1e-9
#define MAX_ITER 1000


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


int **out_knn, **in_knn;
int *k_out, *k_in;
int *dam1;


double theory_p;
double P_kin[K_MAX + 1];
double P_kout[K_MAX + 1];
double W_plus[MAX_R_CONFIG + 1];
double W_minus[MAX_R_CONFIG + 1];
double S_plus, U_plus;
double S_minus, U_minus;
double R_trusted, R_untrusted;


double CURRENT_GAMMA;
int CURRENT_R;


int run_single_config();


void Constrained_DFS(int u, int rm, int** adj, int* k, int* visited, int* drange);
void generate_powerlaw_sequence_unconstrained(int* sequence, int num_nodes, double gamma_exp, int k_min);
Graph* create_graph_from_sequences(const int* in_degree, const int* out_degree, int num_nodes_param, long long E);
void shuffle(int* array, size_t n);
void free_graph(Graph* g);
void save_pk_comparison(const int* degree_seq, int num_nodes, double gamma, int k_min, const char* filename);


void solve_W_equations_theory();
void calculate_S_U_theory();
void calculate_R_theory();
void perform_theoretical_run();
void run_theoretical_calculation(const int* degree_seq, int num_nodes);


int main(int argc, char** argv) {
    srand(time(NULL));


    int max_r_in_list = 0;
    for (int j = 0; j < NUM_R; j++) {
        if (R_LIST[j] > max_r_in_list) {
            max_r_in_list = R_LIST[j];
        }
    }
    if (max_r_in_list > MAX_R_CONFIG) {
        fprintf(stderr, "Error: The maximum value in R_LIST (%d) exceeds MAX_R_CONFIG (%d). Please increase the value of MAX_R_CONFIG.\n", max_r_in_list, MAX_R_CONFIG);
        return 1;
    }


    for (int i = 0; i < NUM_GAMMA; i++) {
        for (int j = 0; j < NUM_R; j++) {
            CURRENT_GAMMA = GAMMA_LIST[i];
            CURRENT_R = R_LIST[j];
            printf("\n\n###########################################################\n");
            printf("###   STARTING NEW CONFIG: GAMMA = %.2f, R = %d   ###\n", CURRENT_GAMMA, CURRENT_R);
            printf("###########################################################\n\n");
            run_single_config();
        }
    }
    printf("All parameter combinations have been executed.\n");
    return 0;
}


int run_single_config() {

    printf("============================================\n");
    printf(">> Step 1: Generating a unified power-law degree sequence for simulation\n");
    printf("   (N=%d, R=%d)\n", N, CURRENT_R);
    printf("   Degree:  gamma=%.2f, k_min=%d\n", CURRENT_GAMMA, K_MIN);

    int* degree_sf = (int*)malloc(N * sizeof(int));
    if (!degree_sf) { fprintf(stderr, "Error: Failed to allocate memory for degree sequence.\n"); return 1; }
    generate_powerlaw_sequence_unconstrained(degree_sf, N, CURRENT_GAMMA, K_MIN);

    int* in_degree_sf = (int*)malloc(N * sizeof(int));
    int* out_degree_sf = (int*)malloc(N * sizeof(int));
    if (!in_degree_sf || !out_degree_sf) { fprintf(stderr, "Error: Memory allocation failed when copying degree sequences.\n"); return 1; }
    memcpy(in_degree_sf, degree_sf, N * sizeof(int));
    memcpy(out_degree_sf, degree_sf, N * sizeof(int));

    long long E = 0;
    for(int i = 0; i < N; i++) E += degree_sf[i];

    printf("   Unified degree sequence generated, total edges E = %lld (average degree c = %.2f)\n", E, (double)E/N);
    printf("   (Since in-degree/out-degree sequences are identical, sum_in == sum_out is automatically satisfied, no balancing needed)\n");


    char pk_comp_filename[120];
    snprintf(pk_comp_filename, sizeof(pk_comp_filename), "comparison_pk_G%.2f_R%d_kmin%d.txt", CURRENT_GAMMA, CURRENT_R, K_MIN);
    save_pk_comparison(in_degree_sf, N, CURRENT_GAMMA, K_MIN, pk_comp_filename);


    printf("\n============================================\n");
    printf(">> Step 2: Performing theoretical calculation based on the [actual] degree distribution\n");
    run_theoretical_calculation(degree_sf, N);

    free(degree_sf);


    printf("\n============================================\n");
    printf(">> Step 3: Starting simulation (Total %d runs)\n", N_RUN_MAX);

    k_out = (int*)calloc(N, sizeof(int));
    k_in = (int*)calloc(N, sizeof(int));
    dam1 = (int*)calloc(N, sizeof(int));
    out_knn = (int**)calloc(N, sizeof(int*));
    in_knn = (int**)calloc(N, sizeof(int*));
    long *EGSCC_T_m = (long*)calloc(100, sizeof(long));
    long *EGSCC_UT_m = (long*)calloc(100, sizeof(long));
    long *GIN_T_m = (long*)calloc(100, sizeof(long));
    long *GIN_UT_m = (long*)calloc(100, sizeof(long));
    long *GOUT_T_m = (long*)calloc(100, sizeof(long));
    long *GOUT_UT_m = (long*)calloc(100, sizeof(long));

    for (int nrun = 0; nrun < N_RUN_MAX; nrun++) {
        printf(" -> Running simulation %d/%d...\n", nrun + 1, N_RUN_MAX);

        shuffle(in_degree_sf, N);
        shuffle(out_degree_sf, N);

        Graph* g = create_graph_from_sequences(in_degree_sf, out_degree_sf, N, E);
        if (g == NULL) { continue; }

        if (nrun > 0) {
            for(int i = 0; i < N; i++) { if(out_knn[i]) free(out_knn[i]); if(in_knn[i]) free(in_knn[i]); }
        }
        memset(k_out, 0, N * sizeof(int));
        memset(k_in, 0, N * sizeof(int));
        for (int i = 0; i < N; i++) {
            k_out[i] = g->nodes[i].count;
            out_knn[i] = (int*)malloc((k_out[i] > 0 ? k_out[i] : 1) * sizeof(int));
            memcpy(out_knn[i], g->nodes[i].neighbors, k_out[i] * sizeof(int));
        }
        for (int i = 0; i < N; i++) { for (int j = 0; j < k_out[i]; j++) k_in[out_knn[i][j]]++; }
        for (int i = 0; i < N; i++) in_knn[i] = (int*)malloc((k_in[i] > 0 ? k_in[i] : 1) * sizeof(int));
        int* temp_k_in = (int*)calloc(N, sizeof(int));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < k_out[i]; j++) {
                in_knn[out_knn[i][j]][temp_k_in[out_knn[i][j]]++] = i;
            }
        }
        free(temp_k_in);
        free_graph(g);

        float* xd1 = (float*)malloc(N * sizeof(float));
        for(int i=0; i<N; i++) xd1[i] = ((double)rand() / RAND_MAX);

        for (int nc = 0; nc < 100; nc++) {
            float f = nc * 0.01;
            for (int i = 0; i < N; i++) dam1[i] = (xd1[i] < f) ? 0 : 1;

            int max_GC = 0;
            int *egscc_nodes = (int*)calloc(N, sizeof(int));
            int egscc_node_count = 0;
            int* component_map = (int*)calloc(N, sizeof(int));

            for (int n = 0; n < N; n++) {
                if (dam1[n] == 1 && component_map[n] == 0) {
                    int* visited_fwd = (int*)calloc(N, sizeof(int));
                    int* drange_fwd = (int*)malloc(N * sizeof(int));
                    for(int i=0; i<N; i++) drange_fwd[i] = CURRENT_R + 1;
                    Constrained_DFS(n, 1, out_knn, k_out, visited_fwd, drange_fwd);

                    int* visited_bwd = (int*)calloc(N, sizeof(int));
                    int* drange_bwd = (int*)malloc(N * sizeof(int));
                    for(int i=0; i<N; i++) drange_bwd[i] = CURRENT_R + 1;
                    Constrained_DFS(n, 1, in_knn, k_in, visited_bwd, drange_bwd);

                    int current_GC = 0;
                    int *current_scc_nodes_temp = (int*)calloc(N, sizeof(int));
                    int current_scc_count_temp = 0;
                    for (int i = 0; i < N; i++) {
                        if (visited_fwd[i] && visited_bwd[i]) {
                            component_map[i] = 1;
                            current_scc_nodes_temp[current_scc_count_temp++] = i;
                            if (dam1[i] == 1) current_GC++;
                        }
                    }

                    if (current_GC > max_GC) {
                        max_GC = current_GC;
                        egscc_node_count = current_scc_count_temp;
                        memcpy(egscc_nodes, current_scc_nodes_temp, egscc_node_count * sizeof(int));
                    }
                    free(visited_fwd); free(drange_fwd);
                    free(visited_bwd); free(drange_bwd);
                    free(current_scc_nodes_temp);
                }
            }
            int EGSCC_T = 0, EGSCC_UT = 0, GIN_T = 0, GIN_UT = 0, GOUT_T = 0, GOUT_UT = 0;
            if (max_GC > 0) {
                int* visited_gout = (int*)calloc(N, sizeof(int));
                int* drange_gout = (int*)malloc(N * sizeof(int));
                for(int i=0; i<N; i++) drange_gout[i] = CURRENT_R + 1;
                for(int i=0; i<egscc_node_count; i++) if(!visited_gout[egscc_nodes[i]]) Constrained_DFS(egscc_nodes[i], 1, out_knn, k_out, visited_gout, drange_gout);

                int* visited_gin = (int*)calloc(N, sizeof(int));
                int* drange_gin = (int*)malloc(N * sizeof(int));
                for(int i=0; i<N; i++) drange_gin[i] = CURRENT_R + 1;
                for(int i=0; i<egscc_node_count; i++) if(!visited_gin[egscc_nodes[i]]) Constrained_DFS(egscc_nodes[i], 1, in_knn, k_in, visited_gin, drange_gin);

                for(int i=0; i<egscc_node_count; i++) { if(dam1[egscc_nodes[i]] == 1) EGSCC_T++; else EGSCC_UT++; }
                for(int i=0; i<N; i++){
                    if(visited_gout[i]){ if(dam1[i]==1) GOUT_T++; else GOUT_UT++; }
                    if(visited_gin[i]){ if(dam1[i]==1) GIN_T++; else GIN_UT++; }
                }
                free(visited_gout); free(drange_gout);
                free(visited_gin); free(drange_gin);
            }
            EGSCC_T_m[nc] += EGSCC_T; EGSCC_UT_m[nc] += EGSCC_UT;
            GIN_T_m[nc] += GIN_T; GIN_UT_m[nc] += GIN_UT;
            GOUT_T_m[nc] += GOUT_T; GOUT_UT_m[nc] += GOUT_UT;
            free(component_map); free(egscc_nodes);
        }
        free(xd1);
    }


    char sim_filename[120];
    snprintf(sim_filename, sizeof(sim_filename), "SF_SIM_G%.2f_R%d_kmin%d.txt", CURRENT_GAMMA, CURRENT_R, K_MIN);
    FILE *gp_sim = fopen(sim_filename, "w");
    fprintf(gp_sim, "# p_trusted\tEGSCC_T\tEGSCC_UT\tGIN_T\tGIN_UT\tGOUT_T\tGOUT_UT\n");
    for (int nc = 0; nc < 100; nc++) {
        float f = nc * 0.01;
        fprintf(gp_sim, "%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
                1.0 - f,
                (float)EGSCC_T_m[nc] / (float)(N_RUN_MAX * N), (float)EGSCC_UT_m[nc] / (float)(N_RUN_MAX * N),
                (float)GIN_T_m[nc] / (float)(N_RUN_MAX * N), (float)GIN_UT_m[nc] / (float)(N_RUN_MAX * N),
                (float)GOUT_T_m[nc] / (float)(N_RUN_MAX * N), (float)GOUT_UT_m[nc] / (float)(N_RUN_MAX * N));
    }
    fclose(gp_sim);
    printf("\n>> Simulation finished. Results saved to %s\n", sim_filename);


    free(in_degree_sf);
    free(out_degree_sf);
    free(k_out); free(k_in); free(dam1);
    for (int i = 0; i < N; i++) {
        if(out_knn && out_knn[i]) free(out_knn[i]);
        if(in_knn && in_knn[i]) free(in_knn[i]);
    }
    free(out_knn); free(in_knn);
    free(EGSCC_T_m); free(EGSCC_UT_m);
    free(GIN_T_m); free(GIN_UT_m);
    free(GOUT_T_m); free(GOUT_UT_m);

    return 0;
}


void save_pk_comparison(const int* degree_seq, int num_nodes, double gamma, int k_min, const char* filename) {
    printf("   Generating comparison file for theoretical and empirical P(k): %s\n", filename);
    int* counts = (int*)calloc(K_MAX + 1, sizeof(int));
    if (!counts) { fprintf(stderr, "Error: Failed to allocate memory for empirical P(k) counts.\n"); return; }
    for (int i = 0; i < num_nodes; i++) { if (degree_seq[i] <= K_MAX) counts[degree_seq[i]]++; }

    double* theoretical_pk = (double*)calloc(K_MAX + 1, sizeof(double));
    if (!theoretical_pk) { fprintf(stderr, "Error: Failed to allocate memory for theoretical P(k).\n"); free(counts); return; }

    double normalization_constant = 0.0;
    for (int k = k_min; k <= K_MAX; ++k) normalization_constant += pow((double)k, -gamma);
    if (normalization_constant > 0) {
        for (int k = k_min; k <= K_MAX; ++k) theoretical_pk[k] = pow((double)k, -gamma) / normalization_constant;
    }

    FILE* fp = fopen(filename, "w");
    if (!fp) { perror("Error: Unable to open P(k) comparison file"); free(counts); free(theoretical_pk); return; }

    fprintf(fp, "# k\tTheoretical_P(k)\tEmpirical_P(k)\n");
    for (int k = 0; k <= K_MAX; k++) {
        double empirical_pk_val = (double)counts[k] / num_nodes;
        if (theoretical_pk[k] > 1e-12 || empirical_pk_val > 1e-12) {
            fprintf(fp, "%d\t%e\t%e\n", k, theoretical_pk[k], empirical_pk_val);
        }
    }
    fclose(fp);
    free(counts);
    free(theoretical_pk);
    printf("   Comparison file saved successfully.\n");
}

void perform_theoretical_run() {
    char theory_filename[120];
    snprintf(theory_filename, sizeof(theory_filename), "SF_THEORY_G%.2f_R%d_kmin%d.txt", CURRENT_GAMMA, CURRENT_R, K_MIN);
    FILE *fp_theory = fopen(theory_filename, "w");
    if (fp_theory == NULL) {
        perror("Error: Unable to open theoretical results file"); return;
    }

    fprintf(fp_theory, "# p_trusted\tS_plus\tS_minus\tU_plus\tU_minus\tSCGC_T\tSCGC_UT\n");
    printf("   Starting iterative calculation of theoretical values (p from 1.00 to 0.00)...\n");
    for (int i = 100; i >= 0; i--) {
        theory_p = (double)i / 100.0;
        solve_W_equations_theory();
        calculate_S_U_theory();
        calculate_R_theory();
        fprintf(fp_theory, "%.4f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                theory_p, S_plus, S_minus, U_plus, U_minus, R_trusted, R_untrusted);
    }
    fclose(fp_theory);
    printf("   Theoretical calculation completed. Results saved to %s\n", theory_filename);
}

void run_theoretical_calculation(const int* degree_seq, int num_nodes) {
    printf("   Calculating P(k) distribution from the actual sequence...\n");

    memset(P_kin, 0, (K_MAX + 1) * sizeof(double));
    memset(P_kout, 0, (K_MAX + 1) * sizeof(double));

    int* counts = (int*)calloc(K_MAX + 1, sizeof(int));
    if (!counts) { fprintf(stderr, "Error: Failed to allocate memory for empirical P(k) counts.\n"); return; }
    for (int i = 0; i < num_nodes; i++) {
        if (degree_seq[i] <= K_MAX) {
            counts[degree_seq[i]]++;
        }
    }

    for (int k = 0; k <= K_MAX; k++) {
        if (counts[k] > 0) {
            P_kin[k] = (double)counts[k] / num_nodes;
            P_kout[k] = (double)counts[k] / num_nodes;
        }
    }
    free(counts);

    perform_theoretical_run();
}


void solve_W_equations_theory() {
    for (int r = 1; r <= CURRENT_R; ++r) { W_plus[r] = 0.01; W_minus[r] = 0.01; }
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double W_plus_new[MAX_R_CONFIG + 1] = {0}, W_minus_new[MAX_R_CONFIG + 1] = {0};
        double diff = 0.0;
        double sum_W_plus[MAX_R_CONFIG + 1] = {0};
        for (int r = 1; r <= CURRENT_R; ++r) sum_W_plus[r] = sum_W_plus[r - 1] + W_plus[r];
        double term1_plus = 0;
        double term_r_plus[MAX_R_CONFIG + 1] = {0};
        for (int k_in = K_MIN; k_in <= K_MAX; ++k_in) {
            if (P_kin[k_in] == 0) continue;
            term1_plus += P_kin[k_in] * (1.0 - pow(1.0 - sum_W_plus[CURRENT_R], k_in));
            for (int r = 1; r < CURRENT_R; ++r) {
                term_r_plus[r+1] += P_kin[k_in] * (pow(1.0 - sum_W_plus[r-1], k_in) - pow(1.0 - sum_W_plus[r], k_in));
            }
        }
        W_plus_new[1] = theory_p * term1_plus;
        for (int r = 2; r <= CURRENT_R; ++r) W_plus_new[r] = (1 - theory_p) * term_r_plus[r];
        double sum_W_minus[MAX_R_CONFIG + 1] = {0};
        for (int r = 1; r <= CURRENT_R; ++r) sum_W_minus[r] = sum_W_minus[r - 1] + W_minus[r];
        double term1_minus = 0;
        double term_r_minus[MAX_R_CONFIG + 1] = {0};
        for (int k_out = K_MIN; k_out <= K_MAX; ++k_out) {
            if (P_kout[k_out] == 0) continue;
            term1_minus += P_kout[k_out] * (1.0 - pow(1.0 - sum_W_minus[CURRENT_R], k_out));
            for (int r = 1; r < CURRENT_R; ++r) {
                term_r_minus[r+1] += P_kout[k_out] * (pow(1.0 - sum_W_minus[r-1], k_out) - pow(1.0 - sum_W_minus[r], k_out));
            }
        }
        W_minus_new[1] = theory_p * term1_minus;
        for (int r = 2; r <= CURRENT_R; ++r) W_minus_new[r] = (1 - theory_p) * term_r_minus[r];
        for (int r = 1; r <= CURRENT_R; ++r) {
            diff += fabs(W_plus_new[r] - W_plus[r]) + fabs(W_minus_new[r] - W_minus[r]);
            W_plus[r] = W_plus_new[r]; W_minus[r] = W_minus_new[r];
        }
        if (diff < TOLERANCE) return;
    }
}

void calculate_S_U_theory() {
    double sum_W_plus_R = 0, sum_W_minus_R = 0;
    double sum_W_plus_R_minus_1 = 0, sum_W_minus_R_minus_1 = 0;
    for (int r = 1; r <= CURRENT_R; ++r) {
        sum_W_plus_R += W_plus[r]; sum_W_minus_R += W_minus[r];
        if (r < CURRENT_R) { sum_W_plus_R_minus_1 += W_plus[r]; sum_W_minus_R_minus_1 += W_minus[r]; }
    }
    double term_S_p = 0, term_U_p = 0;
    for (int k_in = K_MIN; k_in <= K_MAX; ++k_in) {
        if (P_kin[k_in] == 0) continue;
        term_S_p += P_kin[k_in] * (1.0 - pow(1.0 - sum_W_plus_R, k_in));
        term_U_p += P_kin[k_in] * (1.0 - pow(1.0 - sum_W_plus_R_minus_1, k_in));
    }
    double term_S_m = 0, term_U_m = 0;
    for (int k_out = K_MIN; k_out <= K_MAX; ++k_out) {
        if (P_kout[k_out] == 0) continue;
        term_S_m += P_kout[k_out] * (1.0 - pow(1.0 - sum_W_minus_R, k_out));
        term_U_m += P_kout[k_out] * (1.0 - pow(1.0 - sum_W_minus_R_minus_1, k_out));
    }
    S_plus = theory_p * term_S_p; S_minus = theory_p * term_S_m;
    U_plus = (1 - theory_p) * term_U_p; U_minus = (1 - theory_p) * term_U_m;
}

void calculate_R_theory() {
    double sum_W_plus_R = 0, sum_W_minus_R = 0;
    double sum_W_plus_R_minus_1 = 0, sum_W_minus_R_minus_1 = 0;
    for (int r = 1; r <= CURRENT_R; ++r) {
        sum_W_plus_R += W_plus[r]; sum_W_minus_R += W_minus[r];
        if (r < CURRENT_R) { sum_W_plus_R_minus_1 += W_plus[r]; sum_W_minus_R_minus_1 += W_minus[r]; }
    }
    double term_S_in = 0;
    for (int k_in = K_MIN; k_in <= K_MAX; ++k_in) {
        if(P_kin[k_in] > 0) term_S_in += P_kin[k_in] * (1.0 - pow(1.0 - sum_W_plus_R, k_in));
    }
    double term_S_out = 0;
    for (int k_out = K_MIN; k_out <= K_MAX; ++k_out) {
        if(P_kout[k_out] > 0) term_S_out += P_kout[k_out] * (1.0 - pow(1.0 - sum_W_minus_R, k_out));
    }
    R_trusted = theory_p * term_S_in * term_S_out;
    double term_U_in = 0;
    for (int k_in = K_MIN; k_in <= K_MAX; ++k_in) {
        if(P_kin[k_in] > 0) term_U_in += P_kin[k_in] * (1.0 - pow(1.0 - sum_W_plus_R_minus_1, k_in));
    }
    double term_U_out = 0;
    for (int k_out = K_MIN; k_out <= K_MAX; ++k_out) {
        if(P_kout[k_out] > 0) term_U_out += P_kout[k_out] * (1.0 - pow(1.0 - sum_W_minus_R_minus_1, k_out));
    }
    R_untrusted = (1-theory_p) * term_U_in * term_U_out;
}


void generate_powerlaw_sequence_unconstrained(int* sequence, int num_nodes, double gamma_exp, int k_min) {
    double exponent = 1.0 / (gamma_exp - 1.0);
    for (int i = 0; i < num_nodes; ++i) {
        double u = (double)rand() / (RAND_MAX + 1.0);
        double degree_double = (double)k_min * pow(1.0 - u, -exponent);
        sequence[i] = (int)round(degree_double);
        if (sequence[i] < k_min) {
            sequence[i] = k_min;
        }
    }
}

void Constrained_DFS(int u, int rm, int** adj, int* k, int* visited, int* drange) {
    visited[u] = 1; drange[u] = rm;
    for (int i = 0; i < k[u]; i++) {
        int v = adj[u][i];
        if (dam1[v] == 1) {
            if (visited[v] == 0) Constrained_DFS(v, 1, adj, k, visited, drange);
        } else {
            if (rm < CURRENT_R && drange[v] > rm + 1) Constrained_DFS(v, rm + 1, adj, k, visited, drange);
        }
    }
}

Graph* create_graph_from_sequences(const int* in_degree, const int* out_degree, int num_nodes_param, long long E) {
    if (E <= 0) return NULL;
    int* out_stubs = (int*)malloc(E * sizeof(int));
    int* in_stubs = (int*)malloc(E * sizeof(int));
    char** adj_matrix = (char**)malloc(num_nodes_param * sizeof(char*));
    if (!out_stubs || !in_stubs || !adj_matrix) { free(out_stubs); free(in_stubs); free(adj_matrix); return NULL; }
    for(int i = 0; i < num_nodes_param; ++i) {
        adj_matrix[i] = (char*)calloc(num_nodes_param, sizeof(char));
        if (!adj_matrix[i]) { for(int j = 0; j < i; ++j) free(adj_matrix[j]); free(adj_matrix); free(out_stubs); free(in_stubs); return NULL; }
    }
    long long current_stub = 0;
    for (int i = 0; i < num_nodes_param; ++i) for (int j = 0; j < out_degree[i]; ++j) out_stubs[current_stub++] = i;
    current_stub = 0;
    for (int i = 0; i < num_nodes_param; ++i) for (int j = 0; j < in_degree[i]; ++j) in_stubs[current_stub++] = i;
    shuffle(out_stubs, E);
    shuffle(in_stubs, E);
    Graph* g = (Graph*)malloc(sizeof(Graph));
    g->node_count = num_nodes_param; g->E = 0;
    g->nodes = (AdjNode*)malloc(num_nodes_param * sizeof(AdjNode));
    for (int i = 0; i < num_nodes_param; ++i) {
        g->nodes[i].count = 0; g->nodes[i].capacity = 10;
        g->nodes[i].neighbors = (int*)malloc(g->nodes[i].capacity * sizeof(int));
    }
    for (long long i = 0; i < E; ++i) {
        int u = out_stubs[i]; int v = in_stubs[i];
        if (u != v && !adj_matrix[u][v]) {
            adj_matrix[u][v] = 1;
            AdjNode* node = &g->nodes[u];
            if (node->count >= node->capacity) {
                node->capacity *= 2;
                node->neighbors = (int*)realloc(node->neighbors, node->capacity * sizeof(int));
            }
            node->neighbors[node->count++] = v;
            g->E++;
        }
    }
    free(out_stubs); free(in_stubs);
    for(int i = 0; i < num_nodes_param; ++i) free(adj_matrix[i]);
    free(adj_matrix);
    return g;
}

void shuffle(int* array, size_t n) {
    if (n > 1) {
        for (size_t i = n - 1; i > 0; i--) {
            size_t j = (size_t)rand() % (i + 1);
            int t = array[j]; array[j] = array[i]; array[i] = t;
        }
    }
}

void free_graph(Graph* g) {
    if (!g) return;
    for (int i = 0; i < g->node_count; i++) free(g->nodes[i].neighbors);
    free(g->nodes);
    free(g);
}
