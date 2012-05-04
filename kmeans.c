#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <signal.h>
#include <fcntl.h>
#include <pthread.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/param.h>
#include <sys/sysctl.h>

/* k-means problem instance description */

typedef struct _Problem {

    /* problem description parameters */
    int points,
        dimensions,
        clusters;

    /* data matrix: points x dimensions */
    double *data;

    /* algorithm parameters */
    int population;

} Problem;

/* accessor macros */

#define N p.points
#define D p.dimensions
#define C p.clusters
#define P p.population

/* structures used for solving instances */

/*** WARNING: these have to agree on first fields. ***/

typedef struct _Point {
    double distance;
    int cluster;
    int index;
} Point;

typedef struct _Candidate {
    double distance;
    int cluster;
} Candidate;

#define dist(x)  ((Point*)x)->distance
#define clust(x) ((Point*)x)->cluster
#define idx(x)   ((Point*)x)->index

#define cmp(a,b,eq) ((a) < (b) ? -1 : (a) > (b) ? +1 : (eq))

static int by_distance_down(const void *a, const void *b) {
    return cmp(dist(b), dist(a), 0);
}
static int by_distance_up(const void *a, const void *b) {
    return cmp(dist(a), dist(b), 0);
}
static int by_cluster(const void *a, const void *b) {
    return cmp(clust(a), clust(b), by_distance_down(a,b));
}
static int by_index(const void *a, const void *b) {
    return cmp(idx(a), idx(b), 0);
}
static int by_coordinate(const void *a, const void *b) {
    return cmp(**((double**)a), **((double**)b), 0);
}

/* some utility macros */

#define sqr(x)          ((x)*(x))
#define tri(k,l)        ((((l)*((k)+1))>>1)+(l))
#define allocate(t,n)   ((t*)(malloc((n)*sizeof(t))))
#define zallocate(t,n)  ((t*)(calloc(n,sizeof(t))))

/* working data for a solution instance */

typedef struct _Solution {

    Point *points;      /* sortable index into data  */
    int *offsets;       /* where each cluster begins */
    double *means;      /* cluster means             */
    double *inter;      /* inter-cluster distances   */
    Candidate *active;  /* active cluster candidates */
    double *total;      /* total squared distance    */

} Solution;

#define total(x) ((Solution*)x)->total

static int by_fitness(const void *a, const void *b) {
    return cmp(total(b), total(a), 0);
}

#define index(i)    s.points[i].index
#define cluster(i)  s.points[i].cluster
#define distance(i) s.points[i].distance
#define data(i,j)   p.data[j*N+s.points[i].index]
#define offset(k)   s.offsets[k]
#define mean(k,j)   s.means[j*C+k]
#define count(k)    (s.offsets[k+1]-s.offsets[k])
#define inter(k,l)  s.inter[k >= l ? tri(k,l) : tri(l,k)]
#define global(k,i) (s.offsets[k]+i)

Solution allocate_solution(const Problem p) {
    Solution s;
    s.points  = allocate(Point, N);
    s.offsets = allocate(int, C+1);
    s.means   = allocate(double, C*D);
    s.inter   = allocate(double, tri(C,C)+1);
    s.active  = allocate(Candidate, C);
    s.total   = allocate(double, 1);

    offset(0) = 0;
    offset(C) = N;

    int i;
    for (i = 0; i < N; i++) {
        distance(i) = INFINITY;
        cluster(i) = -1;
        index(i) = i;
    }

    return s;
}

static void free_solution(const Solution s) {
    free(s.points);
    free(s.offsets);
    free(s.means);
    free(s.inter);
    free(s.active);
    free(s.total);
}

static void compute_total(const Problem p, const Solution s) {
    int i;
    s.total[0] = 0.0;
    for (i = 0; i < N; i++) s.total[0] += distance(i);
}

void rand_state(unsigned short state[3]) {
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd == -1) {
        perror("open(/dev/urandom)");
        exit(1);
    }
    int n = read(fd, state, 3*sizeof(unsigned short));
    if (n == -1) {
        perror("read(/dev/urandom)");
        exit(1);
    }
    if (n < 3*sizeof(unsigned short)) {
        fprintf(stderr, "Too few bytes from /dev/urandom.\n");
        exit(1);
    }
    n = close(fd);
    if (n == -1) {
        perror("close(/dev/urandom)");
        exit(1);
    }
}

static double init_centroids(const Problem p, const Solution s) {
    int i, j, k = 0;
    unsigned short state[3];
    rand_state(state);

    /* choose first centroid randomly */
    i = nrand48(state) % N;
    for (j = 0; j < D; j++)
        mean(0,j) = data(i,j);

    for (;;) {
        /* compute closest thus-far-chosen centroid to each point  *
         * and compute total squared distance of points to closest */
        double total = 0.0;
        for (i = 0; i < N; i++) {
            double dist = 0.0;
            for (j = 0; j < D; j++) {
                double d = mean(k,j) - data(i,j);
                dist += d*d; /* Euclidean */
            }
            if (dist < distance(i)) {
                distance(i) = dist;
                cluster(i) = k;
            }
            total += distance(i);
        }
        /* choose next centroid proportional to squared distance */
        if (++k < C) {
            double r = erand48(state)*total;
            double c = 0.0;
            for (i = 0; i < N; i++) {
                c += distance(i);
                if (r <= c) break;
            }
            for (j = 0; j < D; j++)
                mean(k,j) = data(i,j);
        } else {
            return total;
        }
    }
}

static void sort_by_cluster(const Problem p, const Solution s) {
    qsort(s.points, N, sizeof(Point), by_cluster);
}
static void sort_by_index(const Problem p, const Solution s) {
    qsort(s.points, N, sizeof(Point), by_index);
}

static void compute_offsets(const Problem p, const Solution s) {
    offset(0) = 0;
    int i, k, last = 0;
    for (i = 0; i < N; i++) {
        if (cluster(i) != last) {
            for (k = last+1; k <= cluster(i); k++) offset(k) = i;
            last = cluster(i);
        }
    }
    for (k = last+1; k < C; k++) offset(k) = N;
}

static void compute_means(const Problem p, const Solution s) {
    unsigned short state[3];
    rand_state(state);

    for (;;) {
        int i, j, k;
        bzero(s.means, C*D*sizeof(double));
        for (j = 0; j < D; j++) {
            for (i = 0; i < N; i++) mean(cluster(i),j) += data(i,j);
            for (k = 0; k < C; k++) mean(k,j) /= count(k);
        }
        /* handle centroids with zero points */
        int clean = 1;
        for (k = 0; k < C; k++) {
            if (count(k)) continue;
            double r = erand48(state)*s.total[0];
            double c = 0.0;
            for (i = 0; i < N; i++) {
                c += distance(i);
                if (r <= c) break;
            }
            for (j = 0; j < D; j++) mean(k,j) = data(i,j);
            s.total[0] -= distance(i);
            distance(i) = 0.0;
            cluster(i) = k;
            clean = 0;
        }
        if (clean) return;
        sort_by_cluster(p,s);
        compute_offsets(p,s);
    }
}

static void compute_inter_cluster(const Problem p, const Solution s) {
    int j, k, l;
    bzero(s.inter, (tri(C,C)+1)*sizeof(double));
    for (j = 0; j < D; j++) {
        for (k = 0; k < C; k++) {
            double m = mean(k,j);
            for (l = 0; l < k; l++) {
                double d = 0.5*(m - mean(l,j));
                inter(k,l) += sqr(d); /* Euclidean */
            }
        }
    }
}

static int reassociate_points(const Problem p, const Solution s) {
    int i, j, k, l, changed = 0;
    for (k = 0; k < C; k++) {
        int M = count(k);
        Point *base = s.points + offset(k);
        qsort(base, M, sizeof(Point), by_distance_down);
        for (l = 0; l < C; l++) {
            s.active[l].distance = inter(k,l);
            s.active[l].cluster = l;
        }
        s.active[k].distance = INFINITY;
        qsort(s.active, C, sizeof(Candidate), by_distance_up);
        for (i = 0; i < M; i++) {
            int g = global(k,i);
            for (l = 0; s.active[l].distance < INFINITY; l++) {
                int c = s.active[l].cluster;
                if (distance(g) < inter(k,c)) {
                    s.active[l].distance = INFINITY;
                    break;
                }
                double dist = 0.0;
                for (j = 0; j < D; j++) {
                    double d = data(g,j) - mean(c,j);
                    dist += sqr(d); /* Euclidean */
                }
                if (dist < distance(g)) {
                    distance(g) = dist;
                    cluster(g) = c;
                }
            }
            if (cluster(g) != k) changed++;
            if (s.active[0].distance == INFINITY) break;
        }
    }
    return changed;
}

static void record_clusters(const Problem p, const Solution s, int *const clusters) {
    int i, k;

    /* relabel clusters by minimum index */
    for (k = 0; k < C; k++) {
        s.active[k].distance = INFINITY; /* co-opted for index */
        s.active[k].cluster = k;
    }
    for (i = 0; i < N; i++) {
        if (s.active[cluster(i)].distance > index(i))
            s.active[cluster(i)].distance = index(i);
    }
    qsort(s.active, C, sizeof(Candidate), by_distance_up);
    for (k = 0; k < C; k++) s.active[k].distance = k;
    qsort(s.active, C, sizeof(Candidate), by_cluster);

    /* record the final cluster assignments */
    for (i = 0; i < N; i++)
        clusters[index(i)] = s.active[cluster(i)].distance;
}

static void sort_by_fitness(const Problem p, Solution *const s) {
    qsort(s, P, sizeof(Solution), by_fitness);
}

static void compute_point_cluster(const Problem p, const Solution s) {
    int i, j;
    s.total[0] = 0.0;
    for (i = 0; i < N; i++) distance(i) = 0.0;
    for (j = 0; j < D; j++) {
        for (i = 0; i < N; i++) {
            double d = mean(cluster(i),j) - data(i,j);
            distance(i) += sqr(d); /* Euclidean */
            s.total[0] += sqr(d);
        }
    }
}

int core_count() {
#ifdef __APPLE__
    int nm[2];
    size_t len = 4;
    uint32_t count;

    nm[0] = CTL_HW; nm[1] = HW_AVAILCPU;
    sysctl(nm, 2, &count, &len, NULL, 0);

    if(count < 1) {
        nm[1] = HW_NCPU;
        sysctl(nm, 2, &count, &len, NULL, 0);
        if(count < 1) { count = 1; }
    }
    return count;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

typedef struct _Context {
    Problem problem;
    Solution solution;
} Context;

static void print_stats(const Problem p, const Context *const c) {
    if (P == 1) {
        printf("%e\n", *c[0].solution.total/N);
    } else {
        int i;
        double sum = 0.0;
        double min = +INFINITY;
        double max = -INFINITY;
        for (i = 0; i < P; i++) {
            double t = *c[i].solution.total;
            sum += t;
            if (t < min) min = t;
            if (t > max) max = t;
        }
        printf("%e %e %e\n", max/N, sum/P/N, min/N);
    }
}

static void *head(void *c) {
    Problem p = ((Context*)c)->problem;
    Solution s = ((Context*)c)->solution;

    *s.total = init_centroids(p,s);
    compute_point_cluster(p,s);

    pthread_exit(NULL);
}

static void *noop(void *c) { pthread_exit(NULL); }

static void *body(void *c) {
    Problem p = ((Context*)c)->problem;
    Solution s = ((Context*)c)->solution;

    for (;;) {
        sort_by_cluster(p,s);
        compute_offsets(p,s);
        compute_means(p,s);
        compute_inter_cluster(p,s);
        int changed = reassociate_points(p,s);
        compute_point_cluster(p,s);
        if (!changed) break;
    }
    compute_total(p,s);
    sort_by_index(p,s);

    pthread_exit(NULL);
}

static void *tail1(void *c) {
    Problem p = ((Context*)c)->problem;
    Solution s = ((Context*)c)->solution;

    init_centroids(p,s);

    pthread_exit(NULL);
}

static void *tail2(void *c) {
    Problem p = ((Context*)c)->problem;
    Solution s = ((Context*)c)->solution;

    compute_point_cluster(p,s);
    compute_inter_cluster(p,s);
    reassociate_points(p,s);

    pthread_exit(NULL);
}

static int done = 0;

double kmeans(const Problem p, int *const clusters) {
    int i, j, k;

    Problem p_copy = p;
    p_copy.data = allocate(double, N*D);
    Context *ctx = zallocate(Context, P);
    double **means = allocate(double*, P);

    int cores = core_count();
    printf("[cores: %d]\n", cores);
    pthread_t *threads = allocate(pthread_t, cores);

    for (k = 0; k < cores; k++)
        pthread_create(&threads[k], NULL, noop, NULL);

    for (k = 0; k < P; k++) {
        ctx[k].problem = p;
        ctx[k].solution = allocate_solution(p);
        pthread_join(threads[k%cores], NULL);
        pthread_create(&threads[k%cores], NULL, head, &ctx[k]);
    }

    while (!done) {
        for (k = 0; k < P; k++) {
            pthread_join(threads[k%cores], NULL);
            pthread_create(&threads[k%cores], NULL, body, &ctx[k]);
        }
        for (k = 0; k < cores; k++) pthread_join(threads[k], NULL);
        print_stats(p,ctx);
        if (P <= 1) break;

        int changed = 0;
        for (i = 0; i < N; i++) {
            for (j = 0; j < D; j++) {
                for (k = 0; k < P; k++) {
                    Solution s = ctx[k].solution;
                    means[k] = &s.means[j*C+s.points[i].cluster];
                }

                qsort(means, P, sizeof(double*), by_coordinate);
                double median = P & 1 ? *means[(P-1)>>1] :
                                   0.5*(*means[(P-1)>>1] + *means[P>>1]);

                if (p_copy.data[j*N+i] != median) {
                    p_copy.data[j*N+i] = median;
                    changed++;
                }
            }
        }
        if (!changed) break;

        for (k = 0; k < P; k++) {
            ctx[k].problem = p_copy;
            pthread_join(threads[k%cores], NULL);
            pthread_create(&threads[k%cores], NULL, tail1, &ctx[k]);
        }
        for (k = 0; k < P; k++) {
            ctx[k].problem = p;
            pthread_join(threads[k%cores], NULL);
            pthread_create(&threads[k%cores], NULL, tail2, &ctx[k]);
        }
    }
    free(threads);

    record_clusters(p,ctx[P-1].solution,clusters);
    double total = *ctx[P-1].solution.total;
    for (k = 0; k < P; k++) free_solution(ctx[k].solution);
    free(p_copy.data);
    return total;
}

static void sigint_handler(int sig) { done = 1; }

int main(int argc, char **argv) {

    signal(SIGINT, sigint_handler);

    Problem p;

    N = atoi(argv[1]);
    D = atoi(argv[2]);
    C = atoi(argv[3]);
    P = atoi(argv[4]);

    p.data = allocate(double, N*D);
    int *clusters = allocate(int, N);

    double Z = ((double)N)/C;

    unsigned short state[3];
    rand_state(state);

    int i, j;
    for (j = 0; j < D; j++) {
        for (i = 0; i < N; i++) {
            p.data[j*N+i] = (j ? 0.0 : 2.0*floor(i/Z)) + erand48(state);
        }
    }

    kmeans(p, clusters);

    if (N < 1000) {
        for (i = 0; i < N; i++) {
            printf("%2d ", clusters[i]);
            if ((i+1)%((int)floor(Z)) == 0 || i == N-1) printf("\n");
        }
    }
    j = 0;
    for (i = 0; i < N; i++) {
        if (clusters[i] == floor(i/Z)) j++;
    }
    printf("[accuracy: %d%%]\n", 100*j/N);

    return 0;
}
