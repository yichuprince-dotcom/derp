#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define NETWORK_FILE "gnutella04_edges_reindexed.txt"
#define OUTPUT_FILE "RealNet_Decoupled_THEORY.txt"

const int R_LIST[] = {2};

#define PHI_STEP 0.01
#define TOLERANCE 1e-9
#define MAX_ITER 5000


int N = 0;
int MAX_DEG_OUT = 0;
int MAX_DEG_IN = 0;
double* P_out_dist = NULL;
double* P_in_dist = NULL;


void load_network_and_extract_degrees(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error: Cannot open file '%s'.\n", filename);
        exit(1);
    }

    char line[512];
    int max_node = -1;


    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '%' || line[0] == '\n') continue;
        int u, v;
        if (sscanf(line, "%d %d", &u, &v) == 2) {
            if (u > max_node) max_node = u;
            if (v > max_node) max_node = v;
        }
    }

    N = max_node + 1;

    int* out_deg_count = (int*)calloc(N, sizeof(int));
    int* in_deg_count = (int*)calloc(N, sizeof(int));


    fseek(fp, 0, SEEK_SET);
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '%' || line[0] == '\n') continue;
        int u, v;
        if (sscanf(line, "%d %d", &u, &v) == 2) {
            out_deg_count[u]++;
            in_deg_count[v]++;
        }
    }
    fclose(fp);


    for (int i = 0; i < N; ++i) {
        if (out_deg_count[i] > MAX_DEG_OUT) MAX_DEG_OUT = out_deg_count[i];
        if (in_deg_count[i] > MAX_DEG_IN) MAX_DEG_IN = in_deg_count[i];
    }


    P_out_dist = (double*)calloc(MAX_DEG_OUT + 1, sizeof(double));
    P_in_dist = (double*)calloc(MAX_DEG_IN + 1, sizeof(double));


    for (int i = 0; i < N; ++i) {
        P_out_dist[out_deg_count[i]] += 1.0;
        P_in_dist[in_deg_count[i]] += 1.0;
    }
    for (int k = 0; k <= MAX_DEG_OUT; ++k) P_out_dist[k] /= (double)N;
    for (int k = 0; k <= MAX_DEG_IN; ++k) P_in_dist[k] /= (double)N;

    free(out_deg_count);
    free(in_deg_count);

    printf("Network degree extraction complete.\n");
    printf("N = %d\n", N);
    printf("MAX_DEG_OUT = %d, MAX_DEG_IN = %d\n", MAX_DEG_OUT, MAX_DEG_IN);
}


void calc_theoretical_P_inf(double phi, int R, double* P_dist, int max_deg,
                            double* out_P_inf, double* out_P_inf_lim) {
    double W[MAX_ITER + 1];
    double W_next[MAX_ITER + 1];
    double Sum_W[MAX_ITER + 1];

    for(int r = 0; r <= R + 2; ++r) W[r] = 0.5;

    for(int iter = 0; iter < MAX_ITER; ++iter) {
        Sum_W[0] = 0;
        for(int r = 1; r <= R; ++r) Sum_W[r] = Sum_W[r-1] + W[r];

        double G0_vals[20];
        for(int r = 0; r <= R; ++r) {
            double u = 1.0 - Sum_W[r];
            if(u < 0) u = 0;
            double val = 0;
            for(int k = 0; k <= max_deg; ++k) {
                if (k == 0) val += P_dist[k];
                else val += P_dist[k] * pow(u, k);
            }
            G0_vals[r] = val;
        }

        W_next[1] = phi * (1.0 - G0_vals[R]);
        for(int r = 1; r < R; ++r) {
            W_next[r+1] = (1.0 - phi) * (G0_vals[r-1] - G0_vals[r]);
        }

        double err = 0;
        for(int r = 1; r <= R; ++r) {
            err += fabs(W_next[r] - W[r]);
            W[r] = W_next[r];
        }
        if(err < TOLERANCE) break;
    }

    double sum_w = 0, sum_wm1 = 0;
    for(int r = 1; r <= R; ++r) sum_w += W[r];
    for(int r = 1; r < R; ++r) sum_wm1 += W[r];

    double u_inf = 1.0 - sum_w;
    double u_lim = 1.0 - sum_wm1;
    double G0_inf = 0, G0_lim = 0;

    for(int k = 0; k <= max_deg; ++k) {
        if (k == 0) {
            G0_inf += P_dist[k];
            G0_lim += P_dist[k];
        } else {
            G0_inf += P_dist[k] * pow(u_inf, k);
            G0_lim += P_dist[k] * pow(u_lim, k);
        }
    }

    *out_P_inf = 1.0 - G0_inf;
    *out_P_inf_lim = 1.0 - G0_lim;
}


void run_theory_decoupled(int R, double phi,
                          double* th_S_plus, double* th_S_minus,
                          double* th_U_plus, double* th_U_minus,
                          double* th_SCGC_T, double* th_SCGC_UT) {

    double P_inf_out, P_inf_lim_out;
    double P_inf_in, P_inf_lim_in;


    calc_theoretical_P_inf(phi, R, P_out_dist, MAX_DEG_OUT, &P_inf_out, &P_inf_lim_out);
    calc_theoretical_P_inf(phi, R, P_in_dist, MAX_DEG_IN, &P_inf_in, &P_inf_lim_in);


    *th_S_plus = phi * P_inf_out;
    *th_U_plus = (1.0 - phi) * P_inf_lim_out;

    *th_S_minus = phi * P_inf_in;
    *th_U_minus = (1.0 - phi) * P_inf_lim_in;

    *th_SCGC_T = phi * P_inf_out * P_inf_in;
    *th_SCGC_UT = (1.0 - phi) * P_inf_lim_out * P_inf_lim_in;
}


int main() {
    printf("Starting Decoupled Theory Calculation...\n");

    load_network_and_extract_degrees(NETWORK_FILE);

    int num_r = sizeof(R_LIST)/sizeof(int);

    FILE* f = fopen(OUTPUT_FILE, "w");
    if(!f) { perror("File Error"); exit(1); }


    fprintf(f, "R\tPhi\tTh_S_plus\tTh_S_minus\tTh_U_plus\tTh_U_minus\tTh_SCGC_T\tTh_SCGC_UT\n");

    for(int r_idx = 0; r_idx < num_r; ++r_idx) {
        int R = R_LIST[r_idx];
        printf("Computing for R = %d...\n", R);

        for(double phi = 1.0; phi >= -0.0001; phi -= PHI_STEP) {
            if(phi < 0) phi = 0;

            double th_S_plus, th_S_minus, th_U_plus, th_U_minus, th_SCGC_T, th_SCGC_UT;
            run_theory_decoupled(R, phi, &th_S_plus, &th_S_minus, &th_U_plus, &th_U_minus, &th_SCGC_T, &th_SCGC_UT);


            fprintf(f, "%d\t%.2f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                    R, phi, th_S_plus, th_S_minus, th_U_plus, th_U_minus, th_SCGC_T, th_SCGC_UT);
        }
    }

    fclose(f);

    if(P_out_dist) free(P_out_dist);
    if(P_in_dist) free(P_in_dist);

    printf("All calculations done. Results saved to '%s'.\n", OUTPUT_FILE);
    return 0;
}
