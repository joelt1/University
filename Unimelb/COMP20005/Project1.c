/* Submission for COMP20005 Project 1, 2019 Semester 1. Context - computing
   delivery sequences for Alistazon, a start-up company that is planning to use
   drones to deliver parcels to customers.
   
   Authorship Declaration:

   (1) I certify that the program contained in this submission is completely
   my own individual work, except where explicitly noted by comments that
   provide details otherwise.  I understand that work that has been developed
   by another student, or by me in collaboration with other students,
   or by non-students as a result of request, solicitation, or payment,
   may not be submitted for assessment in this subject.  I understand that
   submitting for assessment work developed by or in collaboration with
   other students or non-students constitutes Academic Misconduct, and
   may be penalized by mark deductions, or by other penalties determined
   via the University of Melbourne Academic Honesty Policy, as described
   at https://academicintegrity.unimelb.edu.au.

   (2) I also certify that I have not provided a copy of this work in either
   softcopy or hardcopy or any other form to any other student, and nor will
   I do so until after the marks are released. I understand that providing
   my work to other students, regardless of my intention or any undertakings
   made to me by that other student, is also Academic Misconduct.

   (3) I further understand that providing a copy of the assignment
   specification to any form of code authoring or assignment tutoring
   service, or drawing the attention of others to such services and code
   that may have been made available via such a service, may be regarded
   as Student General Misconduct (interfering with the teaching activities
   of the University and/or inciting others to commit Academic Misconduct).
   I understand that an allegation of Student General Misconduct may arise
   regardless of whether or not I personally make use of such solutions
   or sought benefit from such actions.

   Signed by: Joel Thomas 915951
   Dated:     16/04/2019
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_LINES 999					/* Maximum 999 data lines */
#define FIRST_ROW 0						/* Arrays start from 0 */
#define DRONE_WEIGHT 3.8				/* Drone's weight in kg */
#define MAX_WEIGHT 5.8					/* Maximum drone payload in kg*/
#define DRONE_SPEED 4.2					/* Drone flight speed in m/s*/
#define UNWEIGHTED_RANGE (6300/3.8)		/* Range for unladen drone in m*/
#define X_ORIGIN 0.0					/* Horizontal axis start location in m*/
#define Y_ORIGIN 0.0					/* Vertical axis start location in m*/
#define STAGE_2 2
#define STAGE_3 3
#define STAGE_4 4
#define PRINT_OUTPUT 1					/* Dummy variable to print output */

/* New type used for declaring arrays relating to three columns from data */
typedef double array_t[MAX_LINES];

/* Function prototypes */
int my_getchar();
void discard_header();
int read_arrays(array_t x, array_t y, array_t w);
void S1_output(int num_rows, array_t x, array_t y, array_t w);
double find_distance(double x_start, double y_start, double x_end,
	double y_end);
double delivery_range(double weight);
void check_weight(double weight, int i);
double battery_calc(double distance, double range);
void check_total_battery(double battery_out, double battery_ret, int i);
double battery_usage(int stage, int i, double distance, array_t w,
	int print_out);
double flight_time(double distance);
void print_batteries_dist_time(int stage,
	int batteries, double distance, double time);
void S2_output(int num_rows, array_t x, array_t y, array_t w);
void initialise_array(int array_len, int array[]);
int check_not_complete(int k, int total_packages, int deliveries_completed[]);
void S3_output(int num_rows, array_t x, array_t y, array_t w, double x_start,
	double y_start, int stage);
double compute_centroid(int num_rows, array_t loc);
void S4_output(int num_rows, array_t x, array_t y, array_t w);

/* Combining all functions and forming backbone of program */
int
main(int argc, char *argv[]) {
	int num_rows;							/* Number of data rows in file */
	array_t x_locs, y_locs, weights;		/* Arrays for each column */
	discard_header();
	num_rows = read_arrays(x_locs, y_locs, weights);
	/* Prints output for Stage 1 */
	S1_output(num_rows, x_locs, y_locs, weights);
	printf("\n");
	/* Prints output for Stage 2 */
	S2_output(num_rows, x_locs, y_locs, weights);
	printf("\n");
	/* Prints output for Stage 3 */
	S3_output(num_rows, x_locs, y_locs, weights, X_ORIGIN, Y_ORIGIN, STAGE_3);
	printf("\n");
	/* Prints output for Stage 4 */
	S4_output(num_rows, x_locs, y_locs, weights);
	printf("\n");
	printf("Ta daa!\n");
	return 0;
}

/* Equivalent of getchar() but for cross system usage */
int
my_getchar() {
	int i;
	while ((i=getchar())=='\r') {
	}
	return i;
}

/* Discards header (first row) in data to allow scanf on actual data */ 
void
discard_header() {
	while (my_getchar() != '\n');
}

/* Read data lines into column arrays, return total number of data lines */
int
read_arrays(array_t x, array_t y, array_t w) {
	int row_num = 0;
	double x_value, y_value, w_value;
	while (scanf("%lf%lf%lf", &x_value, &y_value, &w_value) == 3) {
		/* Too many data lines in file, exit program */
		if (row_num == MAX_LINES) {
			printf("Too many data lines, please try again.\n");
			exit(EXIT_FAILURE);
		}
		/* Add each value to relevant column array */
		x[row_num] = x_value;
		y[row_num] = y_value;
		w[row_num] = w_value;
		row_num++;
	}
	return row_num;
}

/* Prints output for Stage 1 */
void
S1_output(int num_rows, array_t x, array_t y, array_t w) {
	int i, row_1 = FIRST_ROW, last_row = num_rows-1;
	double total_weight = 0;
	printf("S1, total data lines: %2d\n", num_rows);
	printf("S1, first data line :  x=%6.1f, y=%6.1f, kg=%4.2f\n",
		x[row_1], y[row_1], w[row_1]);
	printf("S1, final data line :  x=%6.1f, y=%6.1f, kg=%4.2f\n",
		x[last_row], y[last_row], w[last_row]);
	/* Total of the package weights */
	for (i=0; i<num_rows; i++) {
		total_weight += w[i];
	}
	printf("S1, total to deliver: %4.2f kg\n", total_weight);
}

/* Calculates distance to be flown between two points */
double
find_distance(double x_start, double y_start, double x_end, double y_end) {
	/* Formula from specification */
	return sqrt(pow(x_end - x_start, 2) + pow(y_end - y_start, 2));
}

/* Calculates drone delivery range for given weight */
double
delivery_range(double weight) {
	/* Formula from specification */
	return 6300/(DRONE_WEIGHT + weight);
}

/* Ensure package weight does not breach drone's maximum payload */
void
check_weight(double weight, int i) {
	if (weight > MAX_WEIGHT) {
		printf("Package %d delivery weight too large, please try again.\n", i);
		exit(EXIT_FAILURE);
	}
}

/* Calculates battery used for delivery of package as a percentage */
double
battery_calc(double distance, double range) {
	return distance/range*100.0;
}

/* Ensure battery usage for round trip does not exceed 100% */
void
check_total_battery(double battery_out, double battery_ret, int i) {
	if (battery_out + battery_ret > 100.0) {
		printf("Package %d total battery required exceeds 100%%, ", i);
		printf("please try again.\n");
		exit(EXIT_FAILURE);
	}
}

/* Combines above functions and returns battery usage for round trip,
   optionally allowing to print output relevant to each stage */
double
battery_usage(int stage, int i, double distance, array_t w, int print_out) {
	double package_weight, battery_out, battery_ret;
	package_weight = w[i];				/* w[i] refers to i-th package weight */
	check_weight(package_weight, i);
	/* Outgoing battery usage */
	battery_out = battery_calc(distance, delivery_range(package_weight));
	/* Returning battery usage */
	battery_ret = battery_calc(distance, UNWEIGHTED_RANGE);
	check_total_battery(battery_out, battery_ret, i);
	/* Use of dummy variable to optionally print output */
	if (print_out) {
		printf("S%d, package=%3d, distance=%6.1fm, battery out=%4.1f%%, "
			"battery ret=%4.1f%%\n", stage, i, distance, battery_out,
			battery_ret);
	}
	return battery_out + battery_ret;
}

/* Calculate each package's round trip flight time */
double
flight_time(double distance) {
	return 2*distance/DRONE_SPEED;
}

/* Final output for each stage containing information on total batteries, total
   flight distance and total flight time */
void
print_batteries_dist_time(int stage, int batteries, double distance,
		double time) {
	printf("S%d, total batteries required:%4d\n", stage, batteries);
	printf("S%d, total flight distance=%6.1f meters, "
		"total flight time=%4.0f seconds\n", stage, distance, time);
}

/* Prints output for Stage 2 */
void
S2_output(int num_rows, array_t x, array_t y, array_t w) {
	int i, j, total_batteries = 1;
	double distance, distance_next, battery = 100.0, total_distance = 0.0,
	total_time = 0.0;
	/* Stage 2 calculations for every package in data file */
	for (i=0; i<num_rows; i++) {
		distance = find_distance(X_ORIGIN, Y_ORIGIN, x[i], y[i]);
		battery -= battery_usage(STAGE_2, i, distance, w, PRINT_OUTPUT);
		total_distance += 2*distance;
		total_time += flight_time(distance);
		/* Only executes whenever there is another package after the current */
		if (i < num_rows-1) {
			j = i+1;
			distance_next = find_distance(X_ORIGIN, Y_ORIGIN, x[j], y[j]);
			/* Check if need to change battery for next package */
			if (battery < battery_usage(STAGE_2, j, distance_next, w,
					!PRINT_OUTPUT)) {
				printf("S2, change the battery\n");
				/* Reset new battery to full capacity */
				battery = 100.0;
				total_batteries++;
			}
		}
	}
	print_batteries_dist_time(STAGE_2, total_batteries, total_distance,
		total_time);
}

/* Sets every element in given array to -1 */
void
initialise_array(int array_len, int array[]) {
	int k;
	for (k=0; k<array_len; k++) {
		array[k] = -1;
	}
}

/* Ensure package delivery has not been previously completed */
int
check_not_complete(int k, int total_packages, int deliveries_completed[]) {
	int m;
	for (m=0; m<total_packages; m++) {
		/* Delivery already completed */
		if (deliveries_completed[m] == k) {
			return 0;
		}
	}
	/* Delivery not completed yet */
	return 1;
}

/* Prints output for Stage 3 */
void
S3_output(int num_rows, array_t x, array_t y, array_t w, double x_start,
		double y_start, int stage) {
	int i, j, deliveries_completed[num_rows], total_deliveries = 0,
	total_batteries = 1;
	double distance, distance_next, battery = 100.0, total_distance = 0.0,
	total_time = 0.0;
	initialise_array(num_rows, deliveries_completed);
	for (i=0; i<num_rows; i++) {
		/* Delivery not completed yet */
		if (check_not_complete(i, num_rows, deliveries_completed)) {
			distance = find_distance(x_start, y_start, x[i], y[i]);
			battery -= battery_usage(stage, i, distance, w, PRINT_OUTPUT);
			/* Change -1 in position i to package i, delivery now recorded as
			   commpleted */
			deliveries_completed[i] = i;
			total_deliveries++;
			total_distance += 2*distance;
			total_time += flight_time(distance);
			/* Check next packages */
			for (j=i+1; j<num_rows; j++) {
				distance_next = find_distance(x_start, y_start, x[j], y[j]);
				/* Package able to delivered on current battery and not yet
				   completed */
				if (battery > battery_usage(stage, j, distance_next, w,
					 !PRINT_OUTPUT) && check_not_complete(j, num_rows,
					 deliveries_completed)) {
					battery -= battery_usage(stage, j, distance_next, w,
						PRINT_OUTPUT);
					deliveries_completed[j] = j;
					total_deliveries++;
					total_distance += 2*distance_next;
					total_time += flight_time(distance_next);
				}
			}
			/* Finally battery change when above loop doesn't run */
			if (total_deliveries < num_rows) {
				printf("S%d, change the battery\n", stage);
				battery = 100.0;
				total_batteries++;
			}
		}
	}
	print_batteries_dist_time(stage, total_batteries, total_distance,
		total_time);
}

/* Calculate best horizontal or vertical starting coordinate for all packages to
   be delivered from */
double
compute_centroid(int num_rows, array_t loc) {
	int k;
	double sum = 0.0;
	/* Formula from specification */
	for (k=0; k<num_rows; k++) {
		sum+= loc[k];
	}
	return sum/num_rows;
}

/* Prints output for Stage 4 */
void
S4_output(int num_rows, array_t x, array_t y, array_t w) {
	double x_start, y_start;
	/* Horizontal axis start coordinate */
	x_start = compute_centroid(num_rows, x);
	/* Vertical axis start coordinate */
	y_start = compute_centroid(num_rows, y);
	printf("S4, centroid location x=%6.1fm, y=%6.1fm\n", x_start, y_start);
	/* Using previous stage output function to now print output as remainder of
	   function is identical */
	S3_output(num_rows, x, y, w, x_start, y_start, STAGE_4);
}

/* End of Project 1 source code. Programming is fun! (please give me additional
   marks <3) */