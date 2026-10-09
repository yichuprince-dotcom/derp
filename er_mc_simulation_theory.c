#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


const double C_LIST[] = {1.5};
const int NUM_C = sizeof(C_LIST) / sizeof(C_LIST[0]);
const int R_LIST[] = {2, 3, 4};
const int NUM_R = sizeof(R_LIST) / sizeof(R_LIST[0]);


#define N 1000
#define K_MAX 100


#define N_RUN_MAX 100


#define TOLERANCE 1e-9
#define MAX_ITER 10000
#define DELTA_PHI 0.01


int **out_knn, **in_knn;
int *k_out, *k_in;
int *dam1;


double *S_theory, *S_temp_theory;
double P_k[K_MAX + 1];


double CURRENT_C;
int CURRENT_R;


int run_single_config_er_pos();


void run_simulation_er_pos();
void Constrained_DFS(int u, int rm, int** adj, int* k, int* visited, int* drange);
int generate_poisson(double lambda);
void shuffle(int* array, long n);


void perform_theoretical_run_er_pos();
void solve_equations_iteration_er_pos(double phi_trusted, FILE *f);
void precompute_poisson_distribution();
long double factorial(int n);
double g1_er_pos(double x);
double sum_er_pos(double *vec, int a, int b);


int main(int argc, char** argv) {
    srand(time(NULL));

    for (int i = 0; i < NUM_C; i++) {
        for (int j = 0; j < NUM_R; j++) {
            CURRENT_C = C_LIST[i];
            CURRENT_R = R_LIST[j];
            printf("\n\n################################################################\n");
            printf("###   STARTING NEW POS-CORR CONFIG: c = %.2f, R = %d   ###\n", CURRENT_C, CURRENT_R);
            printf("################################################################\n\n");
            run_single_config_er_pos();
        }
    }
    printf("All parameter combinations have been executed.\n");
    return 0;
}

int run_single_config_er_pos() {
    printf("============================================\n");
    printf(">> Step 1: Start theoretical calculation (Corrected Model)\n");
    perform_theoretical_run_er_pos();

    printf("\n============================================\n");
    printf(">> Step 2: Start simulation (Configuration Model)\n");
    run_simulation_er_pos();

    return 0;
}


void perform_theoretical_run_er_pos() {
    char theory_filename[120];
    snprintf(theory_filename, sizeof(theory_filename), "ER_Pos_THEORY_c%.2f_R%d.txt", CURRENT_C, CURRENT_R);
    FILE *fp_theory = fopen(theory_filename, "w");
    if (!fp_theory) {
        perror("Error: Unable to open theoretical results file");
        return;
    }

    fprintf(fp_theory, "# p_trusted\tS_plus\tS_minus\tU_plus\tU_minus\tSCGC_T\tSCGC_UT\n");

    precompute_poisson_distribution();

    S_theory = (double *)calloc(CURRENT_R, sizeof(double));
    S_temp_theory = (double *)calloc(CURRENT_R, sizeof(double));

    printf("   Calculating and saving theoretical results to %s ...\n", theory_filename);
    for (double p = 1.0; p >= 0.0 - 1e-9; p -= DELTA_PHI) {
        solve_equations_iteration_er_pos(p > 0 ? p : 0, fp_theory);
    }

    fclose(fp_theory);
    free(S_theory);
    free(S_temp_theory);
    printf("   Theoretical calculation completed.\n");
}

void solve_equations_iteration_er_pos(double phi_trusted, FILE *f) {
    double err = 1.0;
    unsigned int steps_exact = 0;
    for (int r = 0; r < CURRENT_R; r++) S_theory[r] = 0.1;

    while (err > TOLERANCE) {
        memcpy(S_temp_theory, S_theory, CURRENT_R * sizeof(double));

        double theta_R = sum_er_pos(S_temp_theory, 1, CURRENT_R);
        double one_minus_theta_R = 1.0 - theta_R;
        double term1_sum = 1.0 - one_minus_theta_R * g1_er_pos(one_minus_theta_R);
        S_theory[0] = phi_trusted * term1_sum;

        for (int r = 1; r < CURRENT_R; r++) {
            double theta_r_minus_1 = sum_er_pos(S_temp_theory, 1, r - 1);
            double one_minus_theta_r_minus_1 = 1.0 - theta_r_minus_1;
            double theta_r = sum_er_pos(S_temp_theory, 1, r);
            double one_minus_theta_r = 1.0 - theta_r;
            double term_r_sum = (one_minus_theta_r_minus_1 * g1_er_pos(one_minus_theta_r_minus_1)) -
                                (one_minus_theta_r * g1_er_pos(one_minus_theta_r));
            S_theory[r] = (1.0 - phi_trusted) * term_r_sum;
        }

        err = 0.0;
        for (int i = 0; i < CURRENT_R; i++) err += pow(S_theory[i] - S_temp_theory[i], 2);
        err = sqrt(err);

        if (++steps_exact > MAX_ITER) {
            printf("Warning: Maximum iterations reached at p_trusted=%.2f.\n", phi_trusted);
            break;
        }
    }

    double sum_S_R = sum_er_pos(S_theory, 1, CURRENT_R);
    double sum_S_R_minus_1 = sum_er_pos(S_theory, 1, CURRENT_R - 1);

    double term_S_p = 0.0, term_U_p = 0.0;
    for(int k=0; k<=K_MAX; ++k) {
        if(P_k[k] < 1e-12) continue;
        term_S_p += P_k[k] * (1.0 - pow(1.0 - sum_S_R, k));
        term_U_p += P_k[k] * (1.0 - pow(1.0 - sum_S_R_minus_1, k));
    }
    double S_plus = phi_trusted * term_S_p;
    double U_plus = (1.0 - phi_trusted) * term_U_p;

    double term_SCGC_T = 0, term_SCGC_UT = 0;
    for (int k = 0; k <= K_MAX; ++k) {
         if (P_k[k] < 1e-12) continue;
         term_SCGC_T += P_k[k] * pow(1.0 - pow(1.0 - sum_S_R, k), 2);
         term_SCGC_UT += P_k[k] * pow(1.0 - pow(1.0 - sum_S_R_minus_1, k), 2);
    }
    double SCGC_T = phi_trusted * term_SCGC_T;
    double SCGC_UT = (1.0 - phi_trusted) * term_SCGC_UT;

    fprintf(f, "%.4f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
            phi_trusted, S_plus, S_plus, U_plus, U_plus, SCGC_T, SCGC_UT);
}

void precompute_poisson_distribution() {
    memset(P_k, 0, (K_MAX + 1) * sizeof(double));
    for (int k = 0; k <= K_MAX; k++) {
        long double fact_k = factorial(k);
        if (fact_k > 0) P_k[k] = (pow(CURRENT_C, k) * exp(-CURRENT_C)) / (double)fact_k;
    }
}

long double factorial(int n) {
    if (n < 0) return 0;
    if (n == 0) return 1;
    long double res = 1;
    for (int i = 1; i <= n; ++i) res *= i;
    return res;
}

double g1_er_pos(double x) { return exp(CURRENT_C * (x - 1.0)); }

double sum_er_pos(double *vec, int a, int b) {
  if (a > b) return 0.;
  double x = 0;
  for (int i = a; i <= b; i++) x += vec[i - 1];
  return x;
}


void run_simulation_er_pos() {

    k_out = (int*)calloc(N, sizeof(int));
    k_in = (int*)calloc(N, sizeof(int));
    dam1 = (int*)calloc(N, sizeof(int));
    out_knn = (int**)calloc(N, sizeof(int*));
    in_knn = (int**)calloc(N, sizeof(int*));
    for (int i = 0; i < N; i++) {
        out_knn[i] = (int*)calloc(N, sizeof(int));
        in_knn[i] = (int*)calloc(N, sizeof(int));
    }
    long *EGSCC_T_m = (long*)calloc(101, sizeof(long));
    long *EGSCC_UT_m = (long*)calloc(101, sizeof(long));
    long *GIN_T_m = (long*)calloc(101, sizeof(long));
    long *GIN_UT_m = (long*)calloc(101, sizeof(long));
    long *GOUT_T_m = (long*)calloc(101, sizeof(long));
    long *GOUT_UT_m = (long*)calloc(101, sizeof(long));


    for (int nrun = 0; nrun < N_RUN_MAX; nrun++) {
        printf("   -> Running simulation %d/%d...\n", nrun + 1, N_RUN_MAX);


        int *target_degrees = (int*)calloc(N, sizeof(int));
        long total_degree = 0;
        for (int i = 0; i < N; i++) {
            target_degrees[i] = generate_poisson(CURRENT_C);
            total_degree += target_degrees[i];
        }

        int *out_stubs = (int*)malloc(total_degree * sizeof(int));
        int *in_stubs = (int*)malloc(total_degree * sizeof(int));
        long current_pos = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < target_degrees[i]; j++) {
                out_stubs[current_pos] = i;
                in_stubs[current_pos] = i;
                current_pos++;
            }
        }
        shuffle(in_stubs, total_degree);


        memset(k_out, 0, N * sizeof(int));
        memset(k_in, 0, N * sizeof(int));
        char** final_adj_matrix = (char**)calloc(N, sizeof(char*));
        for(int i = 0; i < N; i++) final_adj_matrix[i] = (char*)calloc(N, sizeof(char));

        for (long l = 0; l < total_degree; l++) {
            int u = out_stubs[l];
            int v = in_stubs[l];
            if (u == v || final_adj_matrix[u][v] == 1 || final_adj_matrix[v][u] == 1) continue;

            if (k_out[u] < N && k_in[v] < N) {
                out_knn[u][k_out[u]++] = v;
                in_knn[v][k_in[v]++] = u;
                final_adj_matrix[u][v] = 1;
            }
        }
        free(out_stubs); free(in_stubs); free(target_degrees);
        for(int i = 0; i < N; i++) free(final_adj_matrix[i]);
        free(final_adj_matrix);


        float* xd1 = (float*)malloc(N * sizeof(float));
        for(int i=0; i<N; i++) xd1[i] = ((double)rand() / RAND_MAX);

        for (int nc = 0; nc <= 100; nc++) {
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
    snprintf(sim_filename, sizeof(sim_filename), "ER_Pos_SIM_c%.2f_R%d.txt", CURRENT_C, CURRENT_R);
    FILE *gp_sim = fopen(sim_filename, "w");
    fprintf(gp_sim, "# p_trusted\tEGSCC_T\tEGSCC_UT\tGIN_T\tGIN_UT\tGOUT_T\tGOUT_UT\n");
    for (int nc = 0; nc <= 100; nc++) {
        float p_trusted = 1.0 - (nc * 0.01);
        fprintf(gp_sim, "%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
                p_trusted,
                (double)EGSCC_T_m[nc] / (double)(N_RUN_MAX * N), (double)EGSCC_UT_m[nc] / (double)(N_RUN_MAX * N),
                (double)GIN_T_m[nc] / (double)(N_RUN_MAX * N), (double)GIN_UT_m[nc] / (double)(N_RUN_MAX * N),
                (double)GOUT_T_m[nc] / (double)(N_RUN_MAX * N), (double)GOUT_UT_m[nc] / (double)(N_RUN_MAX * N));
    }
    fclose(gp_sim);
    printf("\n   Simulation finished. Results saved to %s\n", sim_filename);


    free(k_out); free(k_in); free(dam1);
    for (int i = 0; i < N; i++) {
        free(out_knn[i]);
        free(in_knn[i]);
    }
    free(out_knn); free(in_knn);
    free(EGSCC_T_m); free(EGSCC_UT_m);
    free(GIN_T_m); free(GIN_UT_m);
    free(GOUT_T_m); free(GOUT_UT_m);
}

void Constrained_DFS(int u, int rm, int** adj, int* k, int* visited, int* drange) {
    visited[u] = 1;
    drange[u] = rm;
    for (int i = 0; i < k[u]; i++) {
        int v = adj[u][i];
        if (dam1[v] == 1) {
            if (visited[v] == 0) Constrained_DFS(v, 1, adj, k, visited, drange);
        } else {
            if (rm < CURRENT_R && drange[v] > rm + 1) Constrained_DFS(v, rm + 1, adj, k, visited, drange);
        }
    }
}

int generate_poisson(double lambda) {
    double L = exp(-lambda);
    int k = 0;
    double p = 1.0;
    do {
        k++;
        p *= ((double)rand() / RAND_MAX);
    } while (p > L);
    return k - 1;
}

void shuffle(int* array, long n) {
    if (n > 1) {
        for (long i = n - 1; i > 0; i--) {
            long j = rand() % (i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}
