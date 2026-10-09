#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


#define NETWORK_FILE "gnutella04_edges_reindexed.txt"
#define N_RUN_MAX 100
#define CURRENT_R 2
#define TOLERANCE 1e-5
#define MAX_ITER 10000


int N = 0;
double CURRENT_C = 0.0;


int **out_knn, **in_knn;
int *k_out, *k_in;
int *dam1;


double *S_theory, *S_temp_theory;


void load_network();
void perform_theoretical_run_er();
void solve_equations_iteration_er(double phi_trusted, FILE *f, double p_print);
double g0_er(double x);
double g1_er(double x);
double sum_er(double *vec, int a, int b);
void run_simulation();
void Constrained_DFS(int u, int rm, int** adj, int* k, int* visited, int* drange);


int main(int argc, char** argv) {
    srand(time(NULL));

    printf("\n###########################################################\n");
    printf("###   REAL NETWORK: SIMULATION & ER APPROXIMATION      ###\n");
    printf("###   Theory Step = 0.01, Simulation Step = 0.05        ###\n");
    printf("###########################################################\n\n");


    load_network();


    perform_theoretical_run_er();


    run_simulation();


    free(k_out); free(k_in); free(dam1);
    for (int i = 0; i < N; i++) {
        free(out_knn[i]);
        free(in_knn[i]);
    }
    free(out_knn); free(in_knn);

    printf("\nAll tasks have been executed completely.\n");
    return 0;
}


void load_network() {
    FILE *fp_net = fopen(NETWORK_FILE, "r");
    if (fp_net == NULL) {
        perror("Error: Unable to open network file");
        exit(EXIT_FAILURE);
    }

    int u, v;
    int max_id = -1;
    int edges = 0;


    while (fscanf(fp_net, "%d %d", &u, &v) == 2) {
        if (u > max_id) max_id = u;
        if (v > max_id) max_id = v;
    }
    N = max_id + 1;

    k_out = (int*)calloc(N, sizeof(int));
    k_in  = (int*)calloc(N, sizeof(int));
    dam1  = (int*)calloc(N, sizeof(int));


    rewind(fp_net);
    while (fscanf(fp_net, "%d %d", &u, &v) == 2) {
        if (u != v) {
            k_out[u]++;
            k_in[v]++;
            edges++;
        }
    }


    CURRENT_C = (double)edges / N;
    printf(">> Network loaded: N = %d, Total Edges = %d\n", N, edges);
    printf(">> Calculated Average Degree c = %.4f\n\n", CURRENT_C);


    out_knn = (int**)malloc(N * sizeof(int*));
    in_knn  = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        out_knn[i] = (int*)malloc(k_out[i] * sizeof(int));
        in_knn[i]  = (int*)malloc(k_in[i]  * sizeof(int));
        k_out[i] = 0;
        k_in[i]  = 0;
    }

    rewind(fp_net);
    while (fscanf(fp_net, "%d %d", &u, &v) == 2) {
        if (u != v) {
            out_knn[u][k_out[u]++] = v;
            in_knn[v][k_in[v]++] = u;
        }
    }
    fclose(fp_net);
}


void perform_theoretical_run_er() {
    printf(">> Step 1: Starting theoretical calculation (c = %.4f)...\n", CURRENT_C);
    FILE *fp_theory = fopen("RealNet_ER_THEORY_R2.txt", "w");
    fprintf(fp_theory, "# p_trusted\tS_plus\tS_minus\tU_plus\tU_minus\tSCGC_T\tSCGC_UT\n");

    S_theory = (double *)calloc(CURRENT_R + 1, sizeof(double));
    S_temp_theory = (double *)calloc(CURRENT_R + 1, sizeof(double));


    for (int nc = 0; nc <= 100; nc++) {
        double p = 1.0 - (nc * 0.01);
        for (int i = 1; i <= CURRENT_R; i++) S_theory[i] = 0.5;
        solve_equations_iteration_er(p > 0 ? p : 0, fp_theory, p);
    }

    fclose(fp_theory);
    free(S_theory); free(S_temp_theory);
    printf("   Saved theoretical results to RealNet_ER_THEORY_R2.txt\n\n");
}

void solve_equations_iteration_er(double phi_trusted, FILE *f, double p_print) {
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

        if (++iter_count > MAX_ITER) break;
    }


    double x = (1.0 - g0_er(1.0 - sum_er(S_theory, 1, CURRENT_R)));
    double y = (1.0 - g0_er(1.0 - sum_er(S_theory, 1, CURRENT_R - 1)));
    double S_plus = phi_trusted * x;
    double S_minus = phi_trusted * x;
    double U_plus = (1.0 - phi_trusted) * y;
    double U_minus = (1.0 - phi_trusted) * y;
    double SCGC_T = phi_trusted * x * x;
    double SCGC_UT = (1.0 - phi_trusted) * y * y;

    fprintf(f, "%.2f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
            p_print, S_plus, S_minus, U_plus, U_minus, SCGC_T, SCGC_UT);
}

double g0_er(double x) { return exp(CURRENT_C * (x - 1.0)); }
double g1_er(double x) { return exp(CURRENT_C * (x - 1.0)); }
double sum_er(double *vec, int a, int b) {
    double sum = 0.0;
    if (a > b) return 0.0;
    for (int i = a; i <= b; i++) sum += vec[i];
    return sum;
}


void run_simulation() {
    long *EGSCC_T_m = (long*)calloc(101, sizeof(long));
    long *EGSCC_UT_m = (long*)calloc(101, sizeof(long));
    long *GIN_T_m = (long*)calloc(101, sizeof(long));
    long *GIN_UT_m = (long*)calloc(101, sizeof(long));
    long *GOUT_T_m = (long*)calloc(101, sizeof(long));
    long *GOUT_UT_m = (long*)calloc(101, sizeof(long));

    printf(">> Step 2: Starting %d simulation runs...\n", N_RUN_MAX);
    for (int nrun = 0; nrun < N_RUN_MAX; nrun++) {

        printf("   -> Running simulation %d/%d...\r", nrun + 1, N_RUN_MAX);
        fflush(stdout);

        float* xd1 = (float*)malloc(N * sizeof(float));
        for(int i=0; i<N; i++) xd1[i] = ((double)rand() / RAND_MAX);


        for (int nc = 0; nc <= 100; nc += 5) {
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
    printf("\n");

    FILE *gp_sim = fopen("RealNet_SIM_R2.txt", "w");
    fprintf(gp_sim, "# p_trusted\tEGSCC_T\tEGSCC_UT\tGIN_T\tGIN_UT\tGOUT_T\tGOUT_UT\n");


    for (int nc = 0; nc <= 100; nc++) {
        float p_trusted = 1.0 - (nc * 0.01);

        if (nc % 5 == 0) {
            fprintf(gp_sim, "%.2f\t%f\t%f\t%f\t%f\t%f\t%f\n",
                    p_trusted,
                    (double)EGSCC_T_m[nc] / (double)(N_RUN_MAX * N), (double)EGSCC_UT_m[nc] / (double)(N_RUN_MAX * N),
                    (double)GIN_T_m[nc] / (double)(N_RUN_MAX * N), (double)GIN_UT_m[nc] / (double)(N_RUN_MAX * N),
                    (double)GOUT_T_m[nc] / (double)(N_RUN_MAX * N), (double)GOUT_UT_m[nc] / (double)(N_RUN_MAX * N));
        } else {
            fprintf(gp_sim, "%.2f\tnan\tnan\tnan\tnan\tnan\tnan\n", p_trusted);
        }
    }

    fclose(gp_sim);
    printf("   Saved simulation results to RealNet_SIM_R2.txt\n");

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
