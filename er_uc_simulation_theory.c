#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


const double C_LIST[] = {4.0};
const int NUM_C = sizeof(C_LIST) / sizeof(C_LIST[0]);

const int R_LIST[] = {3};
const int NUM_R = sizeof(R_LIST) / sizeof(R_LIST[0]);


#define N 2000


#define N_RUN_MAX 100


#define TOLERANCE 1e-5
#define MAX_ITER 10000
#define DELTA_PHI 0.01


int **out_knn, **in_knn;
int *k_out, *k_in;
int *dam1;


double *S_theory, *S_temp_theory;


double CURRENT_C;
int CURRENT_R;


int run_single_config_er();


void run_simulation_er();
void Constrained_DFS(int u, int rm, int** adj, int* k, int* visited, int* drange);


void perform_theoretical_run_er();
void solve_equations_iteration_er(double phi_trusted, FILE *f);
double g0_er(double x);
double g1_er(double x);
double sum_er(double *vec, int a, int b);


int main(int argc, char** argv) {
    srand(time(NULL));


    for (int i = 0; i < NUM_C; i++) {
        for (int j = 0; j < NUM_R; j++) {
            CURRENT_C = C_LIST[i];
            CURRENT_R = R_LIST[j];
            printf("\n\n###########################################################\n");
            printf("###   STARTING NEW CONFIG: c = %.2f, R = %d   ###\n", CURRENT_C, CURRENT_R);
            printf("###########################################################\n\n");
            run_single_config_er();
        }
    }
    printf("All parameter combinations have been executed.\n");
    return 0;
}


int run_single_config_er() {

    printf("============================================\n");
    printf(">> Step 1: Start theoretical calculation\n");
    perform_theoretical_run_er();


    printf("\n============================================\n");
    printf(">> Step 2: Start simulation (Total %d runs)\n", N_RUN_MAX);
    run_simulation_er();

    return 0;
}


void perform_theoretical_run_er() {
    char theory_filename[120];
    snprintf(theory_filename, sizeof(theory_filename), "ER_THEORY_c%.2f_R%d.txt", CURRENT_C, CURRENT_R);
    FILE *fp_theory = fopen(theory_filename, "w");
    if (fp_theory == NULL) {
        perror("Error: Unable to open theoretical results file");
        return;
    }


    fprintf(fp_theory, "# p_trusted\tS_plus\tS_minus\tU_plus\tU_minus\tSCGC_T\tSCGC_UT\n");


    S_theory = (double *)calloc(CURRENT_R + 1, sizeof(double));
    S_temp_theory = (double *)calloc(CURRENT_R + 1, sizeof(double));
    if (!S_theory || !S_temp_theory) {
        fprintf(stderr, "Error: Memory allocation for theoretical calculation failed.\n");
        fclose(fp_theory);
        return;
    }

    printf("   Calculating and saving theoretical results to %s ...\n", theory_filename);


    for (double p = 1.0; p >= -1e-9; p -= DELTA_PHI) {

        for (int i = 1; i <= CURRENT_R; i++) {
            S_theory[i] = 0.5;
        }
        solve_equations_iteration_er(p > 0 ? p : 0, fp_theory);
    }

    fclose(fp_theory);
    free(S_theory);
    free(S_temp_theory);
    printf("   Theoretical calculation completed.\n");
}


void solve_equations_iteration_er(double phi_trusted, FILE *f) {
    int r;
    double err = 1.0;
    long iter_count = 0;


    while (err > TOLERANCE) {
        err = 0.0;
        for(r = 1; r <= CURRENT_R; r++) S_temp_theory[r-1] = S_theory[r];


        S_theory[1] = phi_trusted * (1.0 - g1_er(1.0 - sum_er(S_theory, 1, CURRENT_R)));
        for (r = 2; r <= CURRENT_R; r++) {
            S_theory[r] = (1.0 - phi_trusted) * (g1_er(1.0 - sum_er(S_theory, 1, r - 2)) - g1_er(1.0 - sum_er(S_theory, 1, r - 1)));
        }

        for (r = 0; r < CURRENT_R; r++) {
            err += pow(S_theory[r+1] - S_temp_theory[r], 2);
        }
        err = sqrt(err);

        iter_count++;
        if (iter_count > MAX_ITER) {
            fprintf(stderr, "Warning: Maximum iterations reached at p_trusted=%.2f, may not have converged.\n", phi_trusted);
            break;
        }
    }


    double x = (1.0 - g0_er(1.0 - sum_er(S_theory, 1, CURRENT_R)));
    double y = (1.0 - g0_er(1.0 - sum_er(S_theory, 1, CURRENT_R - 1)));
    double S_plus = phi_trusted * x;
    double S_minus = phi_trusted * x;
    double U_plus = (1.0 - phi_trusted) * y;
    double U_minus = (1.0 - phi_trusted) * y;
    double SCGC_T = phi_trusted * x * x;
    double SCGC_UT = (1.0 - phi_trusted) * y * y;

    fprintf(f, "%.4f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
            phi_trusted, S_plus, S_minus, U_plus, U_minus, SCGC_T, SCGC_UT);
}


double g0_er(double x) { return exp(CURRENT_C * (x - 1.0)); }
double g1_er(double x) { return exp(CURRENT_C * (x - 1.0)); }


double sum_er(double *vec, int a, int b) {
    double sum = 0.0;
    if (a > b) return 0.0;
    for (int i = a; i <= b; i++) sum += vec[i];
    return sum;
}


void run_simulation_er() {

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


        memset(k_out, 0, N * sizeof(int));
        memset(k_in, 0, N * sizeof(int));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                if (((double)rand() / RAND_MAX) < CURRENT_C / (double)N) {
                    if (k_out[i] < N && k_in[j] < N) {
                        out_knn[i][k_out[i]++] = j;
                        in_knn[j][k_in[j]++] = i;
                    }
                }
            }
        }


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
    snprintf(sim_filename, sizeof(sim_filename), "ER_SIM_c%.2f_R%d.txt", CURRENT_C, CURRENT_R);
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
            if (visited[v] == 0) {
                Constrained_DFS(v, 1, adj, k, visited, drange);
            }
        } else {
            if (rm < CURRENT_R && drange[v] > rm + 1) {
                Constrained_DFS(v, rm + 1, adj, k, visited, drange);
            }
        }
    }
}
